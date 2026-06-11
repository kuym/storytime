/* Phase-0 parity harness: run Conv1d and a bidirectional LSTM through mlx-c
 * and diff against ONNX Runtime CPU reference tensors (spike/td/*.bin).
 *
 * Tests the two highest-risk kernels for the ONNX->MLX interpreter:
 *   - Conv: ONNX [N,Cin,L] / weight [Cout,Cin,K]  vs  mlx NLC / [Cout,K,Cin]
 *   - LSTM: ONNX bidirectional, gate order iofc (the classic parity trap)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mlx/c/mlx.h"

static mlx_stream S;

static float* read_floats(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
  *n = bytes / sizeof(float);
  float* buf = malloc(bytes);
  if (fread(buf, 1, bytes, f) != (size_t)bytes) { exit(1); }
  fclose(f);
  return buf;
}

static mlx_array load(const char* tag, const int* shape, int ndim) {
  char path[256]; snprintf(path, sizeof path, "spike/td/%s.bin", tag);
  size_t n; float* buf = read_floats(path, &n);
  mlx_array a = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
  free(buf);
  return a;
}

/* RMS + max-abs diff between an mlx array and a reference .bin */
static void compare(const char* name, mlx_array got, const char* ref_tag) {
  /* Force a row-major contiguous copy: transposes are lazy strided views, and
   * mlx_array_data_float32 exposes the underlying (pre-transpose) buffer. */
  mlx_array gc = mlx_array_new();
  mlx_contiguous(&gc, got, /*allow_col_major*/false, S);
  mlx_vector_array v = mlx_vector_array_new();
  mlx_vector_array_append_value(v, gc);
  mlx_eval(v);
  mlx_vector_array_free(v);
  size_t n = mlx_array_size(gc);
  const float* g = mlx_array_data_float32(gc);
  size_t rn; char path[256]; snprintf(path, sizeof path, "spike/td/%s.bin", ref_tag);
  float* r = read_floats(path, &rn);
  if (rn != n) { printf("  %-6s SIZE MISMATCH got=%zu ref=%zu\n", name, n, rn); free(r); return; }
  double se = 0, rs = 0, maxabs = 0;
  for (size_t i = 0; i < n; i++) {
    double d = (double)g[i] - r[i];
    se += d * d; rs += (double)r[i] * r[i];
    if (fabs(d) > maxabs) maxabs = fabs(d);
  }
  double rmse = sqrt(se / n), rrms = sqrt(rs / n);
  printf("  %-6s n=%zu  RMSE=%.3e  maxabs=%.3e  rel=%.3e  (ref_rms=%.3e)\n",
         name, n, rmse, maxabs, rrms > 0 ? rmse / rrms : 0.0, rrms);
  printf("         got[0..4]: %.4f %.4f %.4f %.4f %.4f\n", g[0],g[1],g[2],g[3],g[4]);
  printf("         ref[0..4]: %.4f %.4f %.4f %.4f %.4f\n", r[0],r[1],r[2],r[3],r[4]);
  free(r);
}

static mlx_array T(mlx_array a, const int* axes, size_t k) {
  mlx_array r = mlx_array_new();
  mlx_transpose_axes(&r, a, axes, k, S);
  return r;
}

/* ---- Conv1d ---- */
static void test_conv(void) {
  int sx[3]={1,512,53}, sw[3]={512,512,3}, sb[1]={512};
  mlx_array X = load("conv_X", sx, 3);   /* [N,Cin,L]  */
  mlx_array W = load("conv_W", sw, 3);   /* [Cout,Cin,K] */
  mlx_array B = load("conv_B", sb, 1);
  int ax[3]={0,2,1};
  mlx_array Xnlc = T(X, ax, 3);          /* -> [N,L,Cin]   */
  mlx_array Wokc = T(W, ax, 3);          /* -> [Cout,K,Cin]*/
  mlx_array y = mlx_array_new();
  mlx_conv1d(&y, Xnlc, Wokc, /*stride*/1, /*pad*/1, /*dil*/1, /*groups*/1, S);
  mlx_array yb = mlx_array_new();
  mlx_add(&yb, y, B, S);                 /* broadcast bias over last dim */
  mlx_array yncl = T(yb, ax, 3);         /* back to [N,Cout,L] */
  compare("conv", yncl, "conv_Y");
}

/* ---- helpers for LSTM ---- */
static mlx_array slice1(mlx_array a, int axis, int start, int stop, const int* shape, int ndim) {
  int st[8]={0}, sp[8], sd[8];
  for (int i=0;i<ndim;i++){ sp[i]=shape[i]; sd[i]=1; }
  st[axis]=start; sp[axis]=stop;
  mlx_array r = mlx_array_new();
  mlx_slice(&r, a, st, ndim, sp, ndim, sd, ndim, S);
  return r;
}
static mlx_array reshape2(mlx_array a, int d0, int d1) {
  int s[2]={d0,d1}; mlx_array r=mlx_array_new(); mlx_reshape(&r,a,s,2,S); return r;
}
static mlx_array mm(mlx_array a, mlx_array b){ mlx_array r=mlx_array_new(); mlx_matmul(&r,a,b,S); return r; }
static mlx_array add3(mlx_array a, mlx_array b){ mlx_array r=mlx_array_new(); mlx_add(&r,a,b,S); return r; }
static mlx_array mul(mlx_array a, mlx_array b){ mlx_array r=mlx_array_new(); mlx_multiply(&r,a,b,S); return r; }
static mlx_array sig(mlx_array a){ mlx_array r=mlx_array_new(); mlx_sigmoid(&r,a,S); return r; }
static mlx_array tnh(mlx_array a){ mlx_array r=mlx_array_new(); mlx_tanh(&r,a,S); return r; }

/* ONNX bidirectional LSTM, gate order iofc. Writes Y[seq,dir,hidden] into out. */
static void test_lstm(void) {
  const int seq=8, in=640, hid=256, G=4*256;
  int sx[3]={seq,1,in}, sw[3]={2,G,in}, sr[3]={2,G,hid}, sbb[2]={2,2*G};
  mlx_array X = load("lstm_X", sx, 3);   /* [seq,1,in] */
  mlx_array W = load("lstm_W", sw, 3);
  mlx_array R = load("lstm_R", sr, 3);
  mlx_array Bb= load("lstm_Bb", sbb, 2);
  float* out = calloc((size_t)seq*2*hid, sizeof(float));

  for (int d=0; d<2; d++) {
    /* Wd [G,in], Rd [G,hid], biases */
    mlx_array Wd = reshape2(slice1(W,0,d,d+1,sw,3), G, in);
    mlx_array Rd = reshape2(slice1(R,0,d,d+1,sr,3), G, hid);
    mlx_array Bd = reshape2(slice1(Bb,0,d,d+1,sbb,2), 1, 2*G);
    int ax2[2]={1,0};
    mlx_array WdT = T(Wd, ax2, 2);       /* [in,G]  */
    mlx_array RdT = T(Rd, ax2, 2);       /* [hid,G] */
    int s1[2]={1,2*G};
    mlx_array Wbias = slice1(Bd,1,0,G,s1,2);     /* [1,G] */
    mlx_array Rbias = slice1(Bd,1,G,2*G,s1,2);   /* [1,G] */
    mlx_array bias  = add3(Wbias, Rbias);        /* [1,G] */

    int hs[2]={1,hid};
    float* zeros = calloc(hid, sizeof(float));
    mlx_array H = mlx_array_new_data(zeros, hs, 2, MLX_FLOAT32);
    mlx_array C = mlx_array_new_data(zeros, hs, 2, MLX_FLOAT32);
    free(zeros);

    for (int k=0; k<seq; k++) {
      int t = (d==0) ? k : (seq-1-k);          /* reverse for backward dir */
      mlx_array xt = reshape2(slice1(X,0,t,t+1,sx,3), 1, in);  /* [1,in] */
      mlx_array g = add3(add3(mm(xt,WdT), mm(H,RdT)), bias);   /* [1,G] iofc */
      int gs[2]={1,G};
      mlx_array i = sig(slice1(g,1,0*hid,1*hid,gs,2));
      mlx_array o = sig(slice1(g,1,1*hid,2*hid,gs,2));
      mlx_array f = sig(slice1(g,1,2*hid,3*hid,gs,2));
      mlx_array c = tnh(slice1(g,1,3*hid,4*hid,gs,2));
      C = add3(mul(f,C), mul(i,c));             /* C_t = f*C + i*g */
      H = mul(o, tnh(C));                        /* H_t = o*tanh(C) */
      /* read H_t back */
      mlx_vector_array v = mlx_vector_array_new();
      mlx_vector_array_append_value(v, H); mlx_eval(v); mlx_vector_array_free(v);
      const float* hp = mlx_array_data_float32(H);
      float* dst = out + ((size_t)t*2 + d)*hid;
      for (int j=0;j<hid;j++) dst[j]=hp[j];
    }
  }
  /* Compare to lstm_Y [seq,2,1,hid] (flat seq*2*hid) */
  int ys[3]={seq,2,hid};
  mlx_array got = mlx_array_new_data(out, ys, 3, MLX_FLOAT32);
  free(out);
  compare("lstm", got, "lstm_Y");
}

int main(void) {
  S = mlx_default_cpu_stream_new();
  printf("Phase-0 mlx-c parity vs ONNX Runtime CPU:\n");
  test_conv();
  test_lstm();
  return 0;
}
