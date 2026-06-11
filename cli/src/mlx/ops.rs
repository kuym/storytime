// Op kernels: each ONNX node -> mlx-c calls. Returns (output_name, Val) pairs.
// Ported from the verified spike interpreter (docs/onnx-to-mlx-plan.md).

fn bin(
    env: &Env,
    n: &Node,
    f: unsafe extern "C" fn(*mut mlx_array, mlx_array, mlx_array, mlx_stream) -> i32,
) -> mlx_array {
    let a = ga(env, &n.input[0]);
    let b = ga(env, &n.input[1]);
    op!(f, a, b)
}
fn un(
    env: &Env,
    n: &Node,
    f: unsafe extern "C" fn(*mut mlx_array, mlx_array, mlx_stream) -> i32,
) -> mlx_array {
    let a = ga(env, &n.input[0]);
    op!(f, a)
}

fn run_node(n: &Node, env: &mut Env) -> Vec<(String, Val)> {
    let one = |a: mlx_array| vec![(n.output[0].clone(), Val::A(a))];
    match n.op.as_str() {
        "Identity" => one(ga(env, &n.input[0])),

        // ---- elementwise binary (broadcasting) ----
        "Add" => one(bin(env, n, mlx_add)),
        "Sub" => one(bin(env, n, mlx_subtract)),
        "Mul" => one(bin(env, n, mlx_multiply)),
        "Div" => {
            let a = ga(env, &n.input[0]);
            let b = ga(env, &n.input[1]);
            let int = |x| matches!(dtype(x), v if v == INT64 || v == INT32);
            if int(a) && int(b) {
                // ONNX integer Div truncates toward zero (mlx_divide is float)
                let dt = dtype(a);
                let q = op!(mlx_divide, op!(mlx_astype, a, F32), op!(mlx_astype, b, F32));
                one(op!(mlx_astype, q, dt))
            } else {
                one(op!(mlx_divide, a, b))
            }
        }
        "Pow" => one(bin(env, n, mlx_power)),
        "Equal" => one(bin(env, n, mlx_equal)),
        "Greater" => one(bin(env, n, mlx_greater)),
        "Less" => one(bin(env, n, mlx_less)),
        "And" => one(bin(env, n, mlx_logical_and)),
        "Max" => one(bin(env, n, mlx_maximum)),
        "Min" => one(bin(env, n, mlx_minimum)),

        // ---- elementwise unary ----
        "Sqrt" => one(un(env, n, mlx_sqrt)),
        "Reciprocal" => one(un(env, n, mlx_reciprocal)),
        "Sin" => one(un(env, n, mlx_sin)),
        "Cos" => one(un(env, n, mlx_cos)),
        "Tanh" => one(un(env, n, mlx_tanh)),
        "Exp" => one(un(env, n, mlx_exp)),
        "Atan" => one(un(env, n, mlx_arctan)),
        "Floor" => one(un(env, n, mlx_floor)),
        "Round" => one({
            let a = ga(env, &n.input[0]);
            op!(mlx_round, a, 0)
        }),
        "Sigmoid" => one(un(env, n, mlx_sigmoid)),
        "Not" => one(un(env, n, mlx_logical_not)),
        "Neg" => one(un(env, n, mlx_negative)),

        "LeakyRelu" => {
            let a = ga(env, &n.input[0]);
            let alpha = n.af("alpha", 0.01) as f32;
            let sa = scalar_f32(alpha);
            let scaled = op!(mlx_multiply, a, sa);
            one(op!(mlx_maximum, a, scaled))
        }
        "Clip" => {
            let a = ga(env, &n.input[0]);
            let lo = if has(env, n.input.get(1).map(|s| s.as_str()).unwrap_or("")) {
                ga(env, &n.input[1])
            } else {
                newarr()
            };
            let hi = if n.input.len() > 2 && has(env, &n.input[2]) {
                ga(env, &n.input[2])
            } else {
                newarr()
            };
            one(op!(mlx_clip, a, lo, hi))
        }
        "Where" => {
            let c = ga(env, &n.input[0]);
            let x = ga(env, &n.input[1]);
            let y = ga(env, &n.input[2]);
            one(op!(mlx_where, c, x, y))
        }
        "Cast" => {
            let a = ga(env, &n.input[0]);
            one(op!(mlx_astype, a, onnx_dtype(n.ai("to", 1))))
        }

        // ---- shape / structural ----
        "Shape" => {
            let a = ga(env, &n.input[0]);
            let s: Vec<i64> = shape(a).iter().map(|&d| d as i64).collect();
            let r = n.ai("end", s.len() as i64);
            let st_ = n.ai("start", 0);
            let s = &s[st_ as usize..r as usize];
            one(from_i64(s, &[s.len() as i32]))
        }
        "Reshape" => {
            let a = ga(env, &n.input[0]);
            let tgt = read_i64(ga(env, &n.input[1]));
            let cur = shape(a);
            let total: i64 = cur.iter().map(|&d| d as i64).product();
            let mut out: Vec<i32> = Vec::new();
            let mut infer = None;
            let mut known: i64 = 1;
            for (i, &d) in tgt.iter().enumerate() {
                if d == -1 {
                    infer = Some(i);
                    out.push(0);
                } else if d == 0 {
                    out.push(cur[i]);
                    known *= cur[i] as i64;
                } else {
                    out.push(d as i32);
                    known *= d;
                }
            }
            if let Some(i) = infer {
                out[i] = (total / known) as i32;
            }
            one(reshape(a, &out))
        }
        "Transpose" => {
            let a = ga(env, &n.input[0]);
            let mut perm = n.aints("perm");
            if perm.is_empty() {
                perm = (0..ndim(a) as i64).rev().collect();
            }
            let perm: Vec<i32> = perm.iter().map(|&x| x as i32).collect();
            one(transpose(a, &perm))
        }
        "Concat" => {
            let axis = n.ai("axis", 0) as i32;
            let arrs: Vec<mlx_array> = n.input.iter().map(|i| ga(env, i)).collect();
            let vec = track_vec(unsafe { mlx_vector_array_new_data(arrs.as_ptr(), arrs.len()) });
            one(op!(mlx_concatenate_axis, vec, axis))
        }
        "Unsqueeze" => {
            let a = ga(env, &n.input[0]);
            let mut axes = if n.input.len() > 1 {
                read_i64(ga(env, &n.input[1]))
            } else {
                n.aints("axes")
            };
            let new_rank = ndim(a) as i64 + axes.len() as i64;
            for ax in axes.iter_mut() {
                if *ax < 0 {
                    *ax += new_rank;
                }
            }
            axes.sort();
            let mut s: Vec<i32> = shape(a);
            for ax in axes {
                s.insert(ax as usize, 1);
            }
            one(reshape(a, &s))
        }
        "Squeeze" => {
            let a = ga(env, &n.input[0]);
            let axes = if n.input.len() > 1 {
                read_i64(ga(env, &n.input[1]))
            } else {
                n.aints("axes")
            };
            let cur = shape(a);
            let s: Vec<i32> = if axes.is_empty() {
                cur.iter().cloned().filter(|&d| d != 1).collect()
            } else {
                let norm: Vec<i64> = axes
                    .iter()
                    .map(|&x| if x < 0 { x + cur.len() as i64 } else { x })
                    .collect();
                cur.iter()
                    .enumerate()
                    .filter(|(i, _)| !norm.contains(&(*i as i64)))
                    .map(|(_, &d)| d)
                    .collect()
            };
            one(reshape(a, &s))
        }
        "Expand" => {
            let a = ga(env, &n.input[0]);
            let tgt = read_i64(ga(env, &n.input[1]));
            let cur: Vec<i64> = shape(a).iter().map(|&d| d as i64).collect();
            let r = cur.len().max(tgt.len());
            let mut outs = vec![1i64; r];
            for i in 0..r {
                let c = if i >= r - cur.len() { cur[i - (r - cur.len())] } else { 1 };
                let t = if i >= r - tgt.len() { tgt[i - (r - tgt.len())] } else { 1 };
                outs[i] = c.max(t);
            }
            let oi: Vec<i32> = outs.iter().map(|&x| x as i32).collect();
            one(op!(mlx_broadcast_to, a, oi.as_ptr(), oi.len()))
        }
        "Gather" => one(gather_host(env, n)),
        "ConstantOfShape" => {
            let shp = read_i64(ga(env, &n.input[0]));
            let si: Vec<i32> = shp.iter().map(|&x| x as i32).collect();
            let (val, dt) = if let Some(t) = n.tensor("value") {
                (t, dtype(t))
            } else {
                (scalar_f32(0.0), F32)
            };
            one(op!(mlx_full, si.as_ptr(), si.len(), val, dt))
        }
        "Range" => {
            let start = read_f32(ga(env, &n.input[0]))[0] as f64;
            let limit = read_f32(ga(env, &n.input[1]))[0] as f64;
            let delta = read_f32(ga(env, &n.input[2]))[0] as f64;
            let dt = dtype(ga(env, &n.input[0]));
            one(op!(mlx_arange, start, limit, delta, dt))
        }
        "CumSum" => {
            let a = ga(env, &n.input[0]);
            let axis = read_i64(ga(env, &n.input[1]))[0] as i32;
            one(op!(mlx_cumsum, a, axis, false, true))
        }
        "Softmax" => {
            let a = ga(env, &n.input[0]);
            let r = ndim(a) as i64;
            let mut axis = n.ai("axis", -1);
            if axis < 0 {
                axis += r;
            }
            one(op!(mlx_softmax_axis, a, axis as i32, true))
        }
        "ReduceSum" => reduce(env, n, mlx_sum_axes),
        "ReduceMax" => reduce(env, n, mlx_max_axes),
        "ReduceProd" => reduce(env, n, mlx_prod_axes),

        // ---- linear algebra ----
        "MatMul" => one(bin(env, n, mlx_matmul)),
        "Gemm" => one(gemm(env, n)),
        "Conv" => one(conv(env, n)),
        "ConvTranspose" => one(conv_transpose(env, n)),
        "Resize" => one(resize(env, n)),
        "InstanceNormalization" => one(instance_norm(env, n)),
        "LayerNormalization" => layer_norm(env, n),
        "LSTM" => lstm(env, n),

        "ScatterND" => one(scatter_nd(env, n)),
        "ScatterElements" => one(scatter_elements(env, n)),

        // ---- sequences & control flow ----
        "SequenceEmpty" => vec![(n.output[0].clone(), Val::Seq(Vec::new()))],
        "SequenceAt" => {
            let seq = gseq(env, &n.input[0]);
            let pos = read_i64(ga(env, &n.input[1]))[0];
            let p = if pos < 0 { pos + seq.len() as i64 } else { pos } as usize;
            one(seq[p])
        }
        "SequenceInsert" => {
            let mut seq = gseq(env, &n.input[0]).clone();
            let t = ga(env, &n.input[1]);
            let pos = if n.input.len() > 2 && has(env, &n.input[2]) {
                let p = read_i64(ga(env, &n.input[2]))[0];
                if p < 0 { p + seq.len() as i64 } else { p }
            } else {
                seq.len() as i64
            };
            seq.insert(pos as usize, t);
            vec![(n.output[0].clone(), Val::Seq(seq))]
        }
        "SplitToSequence" => {
            let a = ga(env, &n.input[0]);
            let mut axis = n.ai("axis", 0);
            let r = ndim(a) as i64;
            if axis < 0 {
                axis += r;
            }
            let axis = axis as usize;
            let dim = shape(a)[axis] as i64;
            let split_given = n.input.len() > 1 && has(env, &n.input[1]);
            let keepdims = n.ai("keepdims", 1) != 0;
            let sizes: Vec<i64> = if !split_given {
                vec![1; dim as usize]
            } else {
                let s = ga(env, &n.input[1]);
                let sv = read_i64(s);
                if shape(s).is_empty() {
                    let c = sv[0];
                    let mut v = Vec::new();
                    let mut p = 0;
                    while p < dim {
                        v.push((dim - p).min(c));
                        p += c;
                    }
                    v
                } else {
                    sv
                }
            };
            let mut seq = Vec::new();
            let mut start = 0i64;
            for sz in &sizes {
                let mut chunk = slice_axis(a, axis, start, start + sz);
                if !split_given && !keepdims {
                    let ax = [axis as i32];
                    chunk = op!(mlx_squeeze_axes, chunk, ax.as_ptr(), 1);
                }
                seq.push(chunk);
                start += sz;
            }
            vec![(n.output[0].clone(), Val::Seq(seq))]
        }
        "ConcatFromSequence" => {
            let seq = gseq(env, &n.input[0]);
            let axis = n.ai("axis", 0) as i32;
            let new_axis = n.ai("new_axis", 0);
            let v = track_vec(unsafe { mlx_vector_array_new_data(seq.as_ptr(), seq.len()) });
            if new_axis != 0 {
                one(op!(mlx_stack_axis, v, axis))
            } else {
                one(op!(mlx_concatenate_axis, v, axis))
            }
        }
        "Loop" => loop_op(env, n),
        "If" => {
            let cond = read_i64(ga(env, &n.input[0]))[0] != 0;
            let key = if cond { "then_branch" } else { "else_branch" };
            let br = n.subgraph(key);
            let mut ce = env.clone();
            exec_nodes(&br.nodes, &mut ce);
            n.output
                .iter()
                .zip(br.output.iter())
                .map(|(o, bo)| (o.clone(), ce.get(bo).cloned().unwrap()))
                .collect()
        }

        "TopK" => {
            // last-axis TopK; returns (values in input dtype, indices int64)
            let a = ga(env, &n.input[0]);
            let k = read_i64(ga(env, &n.input[1]))[0] as usize;
            let largest = n.ai("largest", 1) != 0;
            let dims = shape(a);
            let r = dims.len();
            let last = dims[r - 1] as usize;
            let rows: usize = dims[..r - 1].iter().map(|&d| d as usize).product::<usize>().max(1);
            let is_int = dtype(a) == INT64 || dtype(a) == INT32;
            let data = read_f32(a);
            let mut vals = vec![0f32; rows * k];
            let mut idxs = vec![0i64; rows * k];
            for row in 0..rows {
                let seg = &data[row * last..(row + 1) * last];
                let mut order: Vec<usize> = (0..last).collect();
                order.sort_by(|&i, &j| seg[j].partial_cmp(&seg[i]).unwrap().then(i.cmp(&j)));
                if !largest {
                    order.sort_by(|&i, &j| seg[i].partial_cmp(&seg[j]).unwrap().then(i.cmp(&j)));
                }
                for t in 0..k {
                    vals[row * k + t] = seg[order[t]];
                    idxs[row * k + t] = order[t] as i64;
                }
            }
            let mut od = dims.clone();
            od[r - 1] = k as i32;
            let va = if is_int {
                op!(mlx_astype, from_f32(&vals, &od), dtype(a))
            } else {
                from_f32(&vals, &od)
            };
            vec![
                (n.output[0].clone(), Val::A(va)),
                (n.output[1].clone(), Val::A(from_i64(&idxs, &od))),
            ]
        }

        "Pad" => {
            let a = ga(env, &n.input[0]);
            let pads = read_i64(ga(env, &n.input[1]));
            let r = ndim(a);
            let axes: Vec<i32> = (0..r as i32).collect();
            let low: Vec<i32> = (0..r).map(|i| pads[i] as i32).collect();
            let high: Vec<i32> = (0..r).map(|i| pads[r + i] as i32).collect();
            let mode = n.s("mode").unwrap_or_else(|| "constant".into());
            if mode == "reflect" {
                return one(host_pad_reflect(a, &low, &high));
            }
            let pv = if mode == "constant" && n.input.len() > 2 && has(env, &n.input[2]) {
                ga(env, &n.input[2])
            } else {
                scalar_f32(0.0)
            };
            let cmode = std::ffi::CString::new(mode).unwrap();
            one(op!(
                mlx_pad,
                a,
                axes.as_ptr(),
                r,
                low.as_ptr(),
                r,
                high.as_ptr(),
                r,
                pv,
                cmode.as_ptr()
            ))
        }

        // Unseeded RNG: the harmonic+noise vocoder's noise source. Real noise.
        "RandomUniformLike" | "RandomNormalLike" => {
            let shp = shape(ga(env, &n.input[0]));
            if n.op == "RandomNormalLike" {
                let key = rng_key(1);
                let loc = n.af("mean", 0.0) as f32;
                let scale = n.af("scale", 1.0) as f32;
                one(op!(mlx_random_normal, shp.as_ptr(), shp.len(), F32, loc, scale, key))
            } else {
                let key = rng_key(2);
                let low = scalar_f32(n.af("low", 0.0) as f32);
                let high = scalar_f32(n.af("high", 1.0) as f32);
                one(op!(mlx_random_uniform, low, high, shp.as_ptr(), shp.len(), F32, key))
            }
        }

        "Slice" => {
            let a = ga(env, &n.input[0]);
            let starts = read_i64(ga(env, &n.input[1]));
            let ends = read_i64(ga(env, &n.input[2]));
            let r = ndim(a);
            let axes: Vec<i64> = if n.input.len() > 3 && has(env, &n.input[3]) {
                read_i64(ga(env, &n.input[3]))
            } else {
                (0..r as i64).collect()
            };
            let steps: Vec<i64> = if n.input.len() > 4 && has(env, &n.input[4]) {
                read_i64(ga(env, &n.input[4]))
            } else {
                vec![1; axes.len()]
            };
            let dims = shape(a);
            let mut sss: Vec<(i64, i64, i64)> = (0..r).map(|i| (0, dims[i] as i64, 1)).collect();
            for k in 0..axes.len() {
                let mut ax = axes[k];
                if ax < 0 {
                    ax += r as i64;
                }
                let ax = ax as usize;
                let d = dims[ax] as i64;
                let stp = steps[k];
                let (mut s, mut e) = (starts[k], ends[k]);
                if s < 0 {
                    s += d;
                }
                if e < 0 {
                    e += d;
                }
                if stp > 0 {
                    s = s.clamp(0, d);
                    e = e.clamp(0, d);
                } else {
                    s = s.clamp(0, d - 1);
                    e = e.clamp(-1, d - 1);
                }
                sss[ax] = (s, e, stp);
            }
            if sss.iter().any(|&(_, _, st)| st < 0) {
                one(host_slice(a, &sss))
            } else {
                let start: Vec<i32> = sss.iter().map(|&(s, _, _)| s as i32).collect();
                let stop: Vec<i32> = sss.iter().map(|&(_, e, _)| e as i32).collect();
                let stride: Vec<i32> = sss.iter().map(|&(_, _, st)| st as i32).collect();
                one(op!(
                    mlx_slice,
                    a,
                    start.as_ptr(),
                    r,
                    stop.as_ptr(),
                    r,
                    stride.as_ptr(),
                    r
                ))
            }
        }

        other => panic!("MLX backend: unimplemented op '{other}' (node '{}')", n.name),
    }
}

fn reduce(
    env: &Env,
    n: &Node,
    f: unsafe extern "C" fn(*mut mlx_array, mlx_array, *const i32, usize, bool, mlx_stream) -> i32,
) -> Vec<(String, Val)> {
    let a = ga(env, &n.input[0]);
    let r = ndim(a) as i64;
    let mut axes = if n.input.len() > 1 {
        read_i64(ga(env, &n.input[1]))
    } else {
        n.aints("axes")
    };
    if axes.is_empty() {
        axes = (0..r).collect();
    }
    for ax in axes.iter_mut() {
        if *ax < 0 {
            *ax += r;
        }
    }
    let ai: Vec<i32> = axes.iter().map(|&x| x as i32).collect();
    let keep = n.ai("keepdims", 1) != 0;
    let out = op!(f, a, ai.as_ptr(), ai.len(), keep);
    vec![(n.output[0].clone(), Val::A(out))]
}

fn gemm(env: &Env, n: &Node) -> mlx_array {
    let mut a = ga(env, &n.input[0]);
    let mut b = ga(env, &n.input[1]);
    if n.ai("transA", 0) != 0 {
        a = transpose(a, &[1, 0]);
    }
    if n.ai("transB", 0) != 0 {
        b = transpose(b, &[1, 0]);
    }
    let mut y = op!(mlx_matmul, a, b);
    let alpha = n.af("alpha", 1.0) as f32;
    if alpha != 1.0 {
        y = op!(mlx_multiply, y, scalar_f32(alpha));
    }
    if n.input.len() > 2 && has(env, &n.input[2]) {
        let mut c = ga(env, &n.input[2]);
        let beta = n.af("beta", 1.0) as f32;
        if beta != 1.0 {
            c = op!(mlx_multiply, c, scalar_f32(beta));
        }
        y = op!(mlx_add, y, c);
    }
    y
}

fn conv(env: &Env, n: &Node) -> mlx_array {
    let x = ga(env, &n.input[0]); // [N, Cin, L]
    let w = ga(env, &n.input[1]); // [Cout, Cin/g, K]
    let groups = n.ai("group", 1) as i32;
    let strides = n.aints("strides");
    let pads = n.aints("pads");
    let dils = n.aints("dilations");
    let stride = *strides.first().unwrap_or(&1) as i32;
    let dil = *dils.first().unwrap_or(&1) as i32;
    let (plo, phi) = if pads.len() == 2 {
        (pads[0] as i32, pads[1] as i32)
    } else {
        (0, 0)
    };
    let xnlc = transpose(x, &[0, 2, 1]); // [N, L, Cin]
    let wokc = transpose(w, &[0, 2, 1]); // [Cout, K, Cin/g]
    let stride_a = [stride];
    let plo_a = [plo];
    let phi_a = [phi];
    let dil_a = [dil];
    let idil_a = [1i32];
    let y = op!(
        mlx_conv_general,
        xnlc,
        wokc,
        stride_a.as_ptr(),
        1,
        plo_a.as_ptr(),
        1,
        phi_a.as_ptr(),
        1,
        dil_a.as_ptr(),
        1,
        idil_a.as_ptr(),
        1,
        groups,
        false
    );
    let y = if n.input.len() > 2 && has(env, &n.input[2]) {
        op!(mlx_add, y, ga(env, &n.input[2])) // bias broadcasts over last dim (Cout)
    } else {
        y
    };
    transpose(y, &[0, 2, 1]) // back to [N, Cout, L]
}

// ONNX Resize. nearest/asymmetric/floor with integer scale -> repeat_axis.
// linear/half_pixel on the last axis -> host interpolation (constant scale here).
fn resize(env: &Env, n: &Node) -> mlx_array {
    let a = ga(env, &n.input[0]);
    let mode = n.s("mode").unwrap_or_else(|| "nearest".into());
    let dims = shape(a);
    let scales: Vec<f32> = if n.input.len() > 2 && has(env, &n.input[2]) {
        read_f32(ga(env, &n.input[2]))
    } else {
        let sizes = read_i64(ga(env, &n.input[3]));
        sizes.iter().zip(&dims).map(|(&s, &d)| s as f32 / d as f32).collect()
    };
    if mode == "nearest" {
        let mut out = a;
        for ax in 0..scales.len() {
            let rep = scales[ax].round() as i32;
            if rep > 1 {
                out = op!(mlx_repeat_axis, out, rep, ax as i32);
            }
        }
        out
    } else {
        let r = dims.len();
        let last = dims[r - 1] as usize;
        let scale = scales[r - 1];
        let lout = (last as f32 * scale).floor() as usize;
        let rows: usize = dims[..r - 1].iter().map(|&d| d as usize).product::<usize>().max(1);
        let data = read_f32(a);
        let mut out = vec![0f32; rows * lout];
        for row in 0..rows {
            let src = &data[row * last..(row + 1) * last];
            for o in 0..lout {
                let mut sx = (o as f32 + 0.5) / scale - 0.5;
                if sx < 0.0 {
                    sx = 0.0;
                }
                if sx > (last - 1) as f32 {
                    sx = (last - 1) as f32;
                }
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(last - 1);
                let w = sx - x0 as f32;
                out[row * lout + o] = src[x0] * (1.0 - w) + src[x1] * w;
            }
        }
        let mut od = dims.clone();
        od[r - 1] = lout as i32;
        from_f32(&out, &od)
    }
}

fn conv_transpose(env: &Env, n: &Node) -> mlx_array {
    let x = ga(env, &n.input[0]); // [N, Cin, L]
    let w = ga(env, &n.input[1]); // ONNX [Cin, Cout/g, K]
    let groups = n.ai("group", 1) as i32;
    let stride = *n.aints("strides").first().unwrap_or(&1) as i32;
    let dil = *n.aints("dilations").first().unwrap_or(&1) as i32;
    let outpad = *n.aints("output_padding").first().unwrap_or(&0) as i32;
    let pads = n.aints("pads");
    let pad = if pads.is_empty() { 0 } else { pads[0] as i32 };
    let xnlc = transpose(x, &[0, 2, 1]); // [N, L, Cin]
    // ONNX ConvTranspose weight [Cin, Cout/g, K] -> MLX [Cout, K, Cin/g].
    let ws = shape(w);
    let (cin, coutg, k) = (ws[0], ws[1], ws[2]);
    let cing = cin / groups;
    let w4 = reshape(w, &[groups, cing, coutg, k]);
    let wp = transpose(w4, &[0, 2, 3, 1]); // [g, Cout/g, K, Cin/g]
    let wt = reshape(wp, &[groups * coutg, k, cing]); // [Cout, K, Cin/g]
    let y = op!(mlx_conv_transpose1d, xnlc, wt, stride, pad, dil, outpad, groups);
    let y = if n.input.len() > 2 && has(env, &n.input[2]) {
        op!(mlx_add, y, ga(env, &n.input[2]))
    } else {
        y
    };
    transpose(y, &[0, 2, 1])
}

fn instance_norm(env: &Env, n: &Node) -> mlx_array {
    let x = ga(env, &n.input[0]); // [N,C,L]
    let scale = ga(env, &n.input[1]); // [C]
    let bias = ga(env, &n.input[2]); // [C]
    let eps = n.af("epsilon", 1e-5) as f32;
    let axes = [2i32];
    let mean = op!(mlx_mean_axes, x, axes.as_ptr(), 1, true);
    let xc = op!(mlx_subtract, x, mean);
    let sq = op!(mlx_multiply, xc, xc);
    let var = op!(mlx_mean_axes, sq, axes.as_ptr(), 1, true);
    let veps = op!(mlx_add, var, scalar_f32(eps));
    let inv = op!(mlx_rsqrt, veps);
    let norm = op!(mlx_multiply, xc, inv);
    let c = shape(x)[1];
    let scale = reshape(scale, &[1, c, 1]);
    let bias = reshape(bias, &[1, c, 1]);
    let s = op!(mlx_multiply, norm, scale);
    op!(mlx_add, s, bias)
}

fn layer_norm(env: &Env, n: &Node) -> Vec<(String, Val)> {
    let x = ga(env, &n.input[0]);
    let axis = n.ai("axis", -1);
    let r = ndim(x) as i64;
    let ax = if axis < 0 { axis + r } else { axis };
    assert_eq!(ax, r - 1, "only last-axis LayerNorm supported");
    let scale = if has(env, &n.input[1]) { ga(env, &n.input[1]) } else { newarr() };
    let bias = if n.input.len() > 2 && has(env, &n.input[2]) {
        ga(env, &n.input[2])
    } else {
        newarr()
    };
    let eps = n.af("epsilon", 1e-5) as f32;
    let mut r0 = newarr();
    unsafe {
        mlx_fast_layer_norm(&mut r0, x, scale, bias, eps, st());
    }
    vec![(n.output[0].clone(), Val::A(track(r0)))]
}

// ONNX LSTM (forward or bidirectional), gate order iofc.
fn lstm(env: &Env, n: &Node) -> Vec<(String, Val)> {
    let x = ga(env, &n.input[0]); // [seq, batch, in]
    let w = ga(env, &n.input[1]); // [numdir, 4h, in]
    let r = ga(env, &n.input[2]); // [numdir, 4h, h]
    let b = if n.input.len() > 3 && has(env, &n.input[3]) {
        Some(ga(env, &n.input[3]))
    } else {
        None
    }; // [numdir, 8h]
    let sh = shape(x);
    let (seq, batch, inn) = (sh[0], sh[1], sh[2]);
    let hidden = n.ai("hidden_size", shape(r)[2] as i64) as i32;
    let dir = n.s("direction").unwrap_or_else(|| "forward".into());
    let ndir = if dir == "bidirectional" { 2 } else { 1 };
    let g4 = 4 * hidden;

    let xd = read_f32(x);
    let wd = read_f32(w);
    let rd = read_f32(r);
    let bd = b.map(read_f32);

    let mut y = vec![0f32; (seq * ndir * batch * hidden) as usize];
    let mut yh = vec![0f32; (ndir * batch * hidden) as usize];
    let mut yc = vec![0f32; (ndir * batch * hidden) as usize];

    for d in 0..ndir {
        let woff = (d * g4 * inn) as usize;
        let roff = (d * g4 * hidden) as usize;
        let wb = bd.as_ref().map(|_| (d * 8 * hidden) as usize);
        for bt in 0..batch {
            let mut h = vec![0f32; hidden as usize];
            let mut c = vec![0f32; hidden as usize];
            for k in 0..seq {
                let t = if d == 0 { k } else { seq - 1 - k };
                let mut gate = vec![0f32; g4 as usize];
                for gi in 0..g4 {
                    let mut acc = 0f32;
                    let wr = woff + (gi * inn) as usize;
                    for ii in 0..inn {
                        acc += xd[((t * batch + bt) * inn + ii) as usize] * wd[wr + ii as usize];
                    }
                    let rr = roff + (gi * hidden) as usize;
                    for hh in 0..hidden {
                        acc += h[hh as usize] * rd[rr + hh as usize];
                    }
                    if let Some(o) = wb {
                        acc += bd.as_ref().unwrap()[o + gi as usize]
                            + bd.as_ref().unwrap()[o + (g4 + gi) as usize];
                    }
                    gate[gi as usize] = acc;
                }
                let hs = hidden as usize;
                for j in 0..hs {
                    let i = sigmoidf(gate[j]);
                    let o = sigmoidf(gate[hs + j]);
                    let f = sigmoidf(gate[2 * hs + j]);
                    let cc = gate[3 * hs + j].tanh();
                    c[j] = f * c[j] + i * cc;
                    h[j] = o * c[j].tanh();
                }
                for j in 0..hs {
                    let oi = (((t * ndir + d) * batch + bt) * hidden + j as i32) as usize;
                    y[oi] = h[j];
                }
            }
            for j in 0..hidden as usize {
                let oi = ((d * batch + bt) * hidden + j as i32) as usize;
                yh[oi] = h[j];
                yc[oi] = c[j];
            }
        }
    }
    let mut outs = vec![(
        n.output[0].clone(),
        Val::A(from_f32(&y, &[seq, ndir, batch, hidden])),
    )];
    if n.output.len() > 1 && !n.output[1].is_empty() {
        outs.push((n.output[1].clone(), Val::A(from_f32(&yh, &[ndir, batch, hidden]))));
    }
    if n.output.len() > 2 && !n.output[2].is_empty() {
        outs.push((n.output[2].clone(), Val::A(from_f32(&yc, &[ndir, batch, hidden]))));
    }
    outs
}

fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn exec_nodes(nodes: &[Node], env: &mut Env) {
    for n in nodes {
        let outs = run_node(n, env);
        for (k, v) in outs {
            env.insert(k, v);
        }
    }
}

fn slice_axis(a: mlx_array, axis: usize, start: i64, stop: i64) -> mlx_array {
    let r = ndim(a);
    let dims = shape(a);
    let mut st = vec![0i32; r];
    let mut sp: Vec<i32> = dims.clone();
    let sd = vec![1i32; r];
    st[axis] = start as i32;
    sp[axis] = stop as i32;
    op!(mlx_slice, a, st.as_ptr(), r, sp.as_ptr(), r, sd.as_ptr(), r)
}

// ONNX Loop: body(iter, cond, v_carried...) -> (cond_out, v_carried..., scan...).
fn loop_op(env: &Env, n: &Node) -> Vec<(String, Val)> {
    let m = read_i64(ga(env, &n.input[0]))[0];
    let mut cond = read_i64(ga(env, &n.input[1]))[0] != 0;
    let body = n.subgraph("body");
    let ncarry = n.input.len() - 2;
    let mut carried: Vec<Val> = n.input[2..].iter().map(|i| env.get(i).cloned().unwrap()).collect();
    let mut iter = 0i64;
    while iter < m && cond {
        let mut ce = env.clone();
        ce.insert(body.input[0].clone(), Val::A(from_i64(&[iter], &[])));
        ce.insert(body.input[1].clone(), Val::A(bool_scalar(cond)));
        for j in 0..ncarry {
            ce.insert(body.input[2 + j].clone(), carried[j].clone());
        }
        exec_nodes(&body.nodes, &mut ce);
        cond = read_i64(ga(&ce, &body.output[0]))[0] != 0;
        for j in 0..ncarry {
            carried[j] = ce.get(&body.output[1 + j]).cloned().unwrap();
        }
        iter += 1;
    }
    n.output
        .iter()
        .enumerate()
        .map(|(j, o)| (o.clone(), carried[j].clone()))
        .collect()
}

// ONNX Gather = np.take(data, indices, axis). Host implementation.
fn gather_host(env: &Env, n: &Node) -> mlx_array {
    let data = ga(env, &n.input[0]);
    let idx = ga(env, &n.input[1]);
    let ds = shape(data);
    let ish = shape(idx);
    let mut axis = n.ai("axis", 0);
    if axis < 0 {
        axis += ds.len() as i64;
    }
    let axis = axis as usize;
    let dim = ds[axis] as i64;
    let pre: usize = ds[..axis].iter().map(|&d| d as usize).product::<usize>().max(1);
    let post: usize = ds[axis + 1..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let mid: usize = ish.iter().map(|&d| d as usize).product::<usize>().max(1);
    let qv: Vec<i64> = read_i64(idx).iter().map(|&v| if v < 0 { v + dim } else { v }).collect();
    let mut out_shape: Vec<i32> = ds[..axis].to_vec();
    out_shape.extend_from_slice(&ish);
    out_shape.extend_from_slice(&ds[axis + 1..]);
    let dimu = dim as usize;
    let is_int = dtype(data) == INT64 || dtype(data) == INT32;
    if is_int {
        let d = read_i64(data);
        let mut out = vec![0i64; pre * mid * post];
        for p in 0..pre {
            for m in 0..mid {
                let q = qv[m] as usize;
                for t in 0..post {
                    out[(p * mid + m) * post + t] = d[(p * dimu + q) * post + t];
                }
            }
        }
        from_i64(&out, &out_shape)
    } else {
        let d = read_f32(data);
        let mut out = vec![0f32; pre * mid * post];
        for p in 0..pre {
            for m in 0..mid {
                let q = qv[m] as usize;
                for t in 0..post {
                    out[(p * mid + m) * post + t] = d[(p * dimu + q) * post + t];
                }
            }
        }
        from_f32(&out, &out_shape)
    }
}

// Host N-D slice supporting negative steps.
fn host_slice(a: mlx_array, sss: &[(i64, i64, i64)]) -> mlx_array {
    let dims = shape(a);
    let r = dims.len();
    let mut idxs: Vec<Vec<i64>> = Vec::with_capacity(r);
    for i in 0..r {
        let (s, e, st) = sss[i];
        let mut v = Vec::new();
        let mut x = s;
        if st > 0 {
            while x < e {
                v.push(x);
                x += st;
            }
        } else {
            while x > e {
                v.push(x);
                x += st;
            }
        }
        idxs.push(v);
    }
    let out_dims: Vec<i32> = idxs.iter().map(|v| v.len() as i32).collect();
    let out_strides = row_strides(&out_dims);
    let in_strides = row_strides(&dims);
    let total: usize = idxs.iter().map(|v| v.len()).product();
    let map_idx = |li: usize| -> usize {
        let mut flat = 0i64;
        for i in 0..r {
            let coord = (li as i64 / out_strides[i]) % out_dims[i] as i64;
            flat += idxs[i][coord as usize] * in_strides[i];
        }
        flat as usize
    };
    let is_int = dtype(a) == INT64 || dtype(a) == INT32;
    if is_int {
        let d = read_i64(a);
        let out: Vec<i64> = (0..total).map(|li| d[map_idx(li)]).collect();
        from_i64(&out, &out_dims)
    } else {
        let d = read_f32(a);
        let out: Vec<f32> = (0..total).map(|li| d[map_idx(li)]).collect();
        from_f32(&out, &out_dims)
    }
}

// ONNX reflect pad (no edge repeat); mlx_pad doesn't support reflect.
fn host_pad_reflect(a: mlx_array, low: &[i32], high: &[i32]) -> mlx_array {
    let dims = shape(a);
    let r = dims.len();
    let reflect = |x: i64, d: i64| -> i64 {
        if d == 1 {
            return 0;
        }
        let period = 2 * (d - 1);
        let m = ((x % period) + period) % period;
        if m < d {
            m
        } else {
            period - m
        }
    };
    let idxs: Vec<Vec<i64>> = (0..r)
        .map(|i| {
            let d = dims[i] as i64;
            (0..d + low[i] as i64 + high[i] as i64)
                .map(|o| reflect(o - low[i] as i64, d))
                .collect()
        })
        .collect();
    let out_dims: Vec<i32> = idxs.iter().map(|v| v.len() as i32).collect();
    let out_strides = row_strides(&out_dims);
    let in_strides = row_strides(&dims);
    let total: usize = idxs.iter().map(|v| v.len()).product();
    let d = read_f32(a);
    let out: Vec<f32> = (0..total)
        .map(|li| {
            let mut flat = 0i64;
            for i in 0..r {
                let coord = (li as i64 / out_strides[i]) % out_dims[i] as i64;
                flat += idxs[i][coord as usize] * in_strides[i];
            }
            d[flat as usize]
        })
        .collect();
    from_f32(&out, &out_dims)
}

fn row_strides(dims: &[i32]) -> Vec<i64> {
    let mut s = vec![1i64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1] as i64;
    }
    s
}

// ONNX ScatterND (reduction=none). Host implementation.
fn scatter_nd(env: &Env, n: &Node) -> mlx_array {
    let data = ga(env, &n.input[0]);
    let indices = ga(env, &n.input[1]);
    let updates = ga(env, &n.input[2]);
    let ds = shape(data);
    let is_ = shape(indices);
    let k = *is_.last().unwrap() as usize;
    let num: usize = is_[..is_.len() - 1].iter().map(|&d| d as usize).product::<usize>().max(1);
    let stride = row_strides(&ds);
    let block: usize = ds[k..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let idx = read_i64(indices);
    let is_int = dtype(data) == INT64 || dtype(data) == INT32;
    if is_int {
        let mut out = read_i64(data);
        let upd = read_i64(updates);
        for u in 0..num {
            let base: i64 = (0..k).map(|j| idx[u * k + j] * stride[j]).sum();
            for e in 0..block {
                out[base as usize + e] = upd[u * block + e];
            }
        }
        from_i64(&out, &ds)
    } else {
        let mut out = read_f32(data);
        let upd = read_f32(updates);
        for u in 0..num {
            let base: i64 = (0..k).map(|j| idx[u * k + j] * stride[j]).sum();
            for e in 0..block {
                out[base as usize + e] = upd[u * block + e];
            }
        }
        from_f32(&out, &ds)
    }
}

// ONNX ScatterElements (reduction=none). Host implementation.
fn scatter_elements(env: &Env, n: &Node) -> mlx_array {
    let data = ga(env, &n.input[0]);
    let indices = ga(env, &n.input[1]);
    let updates = ga(env, &n.input[2]);
    let ds = shape(data);
    let is_ = shape(indices);
    let axis = {
        let a = n.ai("axis", 0);
        if a < 0 { a + ds.len() as i64 } else { a }
    } as usize;
    let dstride = row_strides(&ds);
    let istride = row_strides(&is_);
    let idx = read_i64(indices);
    let total: usize = is_.iter().map(|&d| d as usize).product::<usize>().max(1);
    let dim = ds[axis] as i64;
    let is_int = dtype(data) == INT64 || dtype(data) == INT32;
    let dest_of = |li: usize| -> usize {
        let mut rem = li;
        let mut dest = 0i64;
        for d in 0..is_.len() {
            let coord = (rem as i64 / istride[d]) % is_[d] as i64;
            rem %= istride[d] as usize;
            let c = if d == axis {
                let mut v = idx[li];
                if v < 0 {
                    v += dim;
                }
                v
            } else {
                coord
            };
            dest += c * dstride[d];
        }
        dest as usize
    };
    if is_int {
        let mut out = read_i64(data);
        let upd = read_i64(updates);
        for li in 0..total {
            out[dest_of(li)] = upd[li];
        }
        from_i64(&out, &ds)
    } else {
        let mut out = read_f32(data);
        let upd = read_f32(updates);
        for li in 0..total {
            out[dest_of(li)] = upd[li];
        }
        from_f32(&out, &ds)
    }
}
