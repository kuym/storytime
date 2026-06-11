// Native ONNX graph parsing + lowering (no protoc: minimal hand-written prost
// messages). Folds Constant nodes into the weight map and materializes every
// initializer as a concrete mlx array (host-backed, so GPU compute can use it
// directly — unlike the lazy safetensors Load path).
use prost::Message;

#[derive(Clone, PartialEq, ::prost::Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeProto>,
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<AttributeProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
struct AttributeProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(float, tag = "2")]
    f: f32,
    #[prost(int64, tag = "3")]
    i: i64,
    #[prost(bytes = "vec", tag = "4")]
    s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    t: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    g: Option<GraphProto>,
    #[prost(float, repeated, tag = "7")]
    floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    ints: Vec<i64>,
    #[prost(int32, tag = "20")]
    r#type: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(float, repeated, tag = "4")]
    float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    int32_data: Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", tag = "9")]
    raw_data: Vec<u8>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
}

// ONNX AttributeProto.AttributeType
const A_FLOAT: i32 = 1;
const A_INT: i32 = 2;
const A_STRING: i32 = 3;
const A_TENSOR: i32 = 4;
const A_GRAPH: i32 = 5;
const A_INTS: i32 = 7;

struct Graph {
    nodes: Vec<Node>,
    weights: HashMap<String, mlx_array>,
}

impl Graph {
    fn load(bytes: &[u8]) -> Result<Graph> {
        let model = ModelProto::decode(bytes).context("decoding ONNX model")?;
        let g = model.graph.context("ONNX model has no graph")?;
        let mut weights = HashMap::new();
        for t in &g.initializer {
            weights.insert(t.name.clone(), tensor_to_mlx(t));
        }
        let nodes = lower_nodes(&g.node, &mut weights);
        Ok(Graph { nodes, weights })
    }
}

fn lower_nodes(nodes: &[NodeProto], weights: &mut HashMap<String, mlx_array>) -> Vec<Node> {
    let mut out = Vec::new();
    for n in nodes {
        if n.op_type == "Constant" {
            if let Some((name, arr)) = fold_constant(n) {
                weights.insert(name, arr);
                continue;
            }
        }
        let mut attr = HashMap::new();
        for a in &n.attribute {
            let v = match a.r#type {
                A_FLOAT => Attr::F(a.f as f64),
                A_INT => Attr::I(a.i),
                A_STRING => Attr::S(String::from_utf8_lossy(&a.s).into_owned()),
                A_TENSOR => Attr::T(tensor_to_mlx(a.t.as_ref().unwrap())),
                A_GRAPH => Attr::G(lower_subgraph(a.g.as_ref().unwrap(), weights)),
                A_INTS => Attr::Ints(a.ints.clone()),
                _ => continue,
            };
            attr.insert(a.name.clone(), v);
        }
        out.push(Node {
            op: n.op_type.clone(),
            name: n.name.clone(),
            input: n.input.clone(),
            output: n.output.clone(),
            attr,
        });
    }
    out
}

fn lower_subgraph(g: &GraphProto, weights: &mut HashMap<String, mlx_array>) -> Subgraph {
    for t in &g.initializer {
        weights.insert(t.name.clone(), tensor_to_mlx(t));
    }
    Subgraph {
        input: g.input.iter().map(|v| v.name.clone()).collect(),
        output: g.output.iter().map(|v| v.name.clone()).collect(),
        nodes: lower_nodes(&g.node, weights),
    }
}

fn fold_constant(n: &NodeProto) -> Option<(String, mlx_array)> {
    let out = n.output.first()?.clone();
    for a in &n.attribute {
        let arr = match a.name.as_str() {
            "value" => a.t.as_ref().map(tensor_to_mlx)?,
            "value_float" => from_f32(&[a.f], &[]),
            "value_int" => from_i64(&[a.i], &[]),
            "value_floats" => from_f32(&a.floats, &[a.floats.len() as i32]),
            "value_ints" => from_i64(&a.ints, &[a.ints.len() as i32]),
            _ => continue,
        };
        return Some((out, arr));
    }
    None
}

fn tensor_to_mlx(t: &TensorProto) -> mlx_array {
    let shp: Vec<i32> = t.dims.iter().map(|&d| d as i32).collect();
    match t.data_type {
        1 => {
            // FLOAT
            if t.raw_data.is_empty() {
                from_f32(&t.float_data, &shp)
            } else {
                let v: Vec<f32> = t
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                from_f32(&v, &shp)
            }
        }
        7 => {
            // INT64
            if t.raw_data.is_empty() {
                from_i64(&t.int64_data, &shp)
            } else {
                let v: Vec<i64> = t
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                from_i64(&v, &shp)
            }
        }
        6 => {
            // INT32
            if t.raw_data.is_empty() {
                from_i32(&t.int32_data, &shp)
            } else {
                let v: Vec<i32> = t
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                from_i32(&v, &shp)
            }
        }
        9 => {
            // BOOL (1 byte each)
            let bytes: Vec<u8> = if t.raw_data.is_empty() {
                t.int32_data.iter().map(|&x| x as u8).collect()
            } else {
                t.raw_data.clone()
            };
            unsafe {
                mlx_array_new_data(
                    bytes.as_ptr() as *const _,
                    shp.as_ptr(),
                    shp.len() as i32,
                    BOOL,
                )
            }
        }
        other => panic!("ONNX tensor dtype {other} not supported"),
    }
}
