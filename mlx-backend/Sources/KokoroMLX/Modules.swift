// Kokoro-82M model components ported from mlx-audio's Python implementation.
// Covers: BiLSTM, CustomAlbert (PLBert), TextEncoder, ProsodyPredictor.

import Foundation
import MLX
import MLXNN

// MARK: - LinearNorm

class LinearNorm: Module {
    let linearLayer: Linear

    init(inDim: Int, outDim: Int, bias: Bool = true) {
        linearLayer = Linear(inDim, outDim, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearLayer(x)
    }
}

// MARK: - Bidirectional LSTM

class BiLSTM: Module {
    let inputSize: Int
    let hiddenSize: Int

    // Property names match weight keys exactly for Mirror-based discovery
    var Wx_forward: MLXArray
    var Wh_forward: MLXArray
    var bias_ih_forward: MLXArray
    var bias_hh_forward: MLXArray
    var Wx_backward: MLXArray
    var Wh_backward: MLXArray
    var bias_ih_backward: MLXArray
    var bias_hh_backward: MLXArray

    init(inputSize: Int, hiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        let s = 1.0 / Float(hiddenSize).squareRoot()
        Wx_forward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize, inputSize])
        Wh_forward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize, hiddenSize])
        bias_ih_forward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize])
        bias_hh_forward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize])
        Wx_backward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize, inputSize])
        Wh_backward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize, hiddenSize])
        bias_ih_backward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize])
        bias_hh_backward = MLXRandom.uniform(low: -s, high: s, [4 * hiddenSize])
    }

    private func forwardDir(_ x: MLXArray) -> MLXArray {
        let xProj = addMM(bias_ih_forward + bias_hh_forward, x, Wx_forward.transposed())
        let seqLen = x.dim(-2)
        var h = MLXArray.zeros([x.dim(0), hiddenSize])
        var c = MLXArray.zeros([x.dim(0), hiddenSize])
        var allH = [MLXArray]()
        for t in 0..<seqLen {
            var ifgo = xProj[.ellipsis, t, 0...]
            ifgo = ifgo + h.matmul(Wh_forward.transposed())
            let parts = ifgo.split(parts: 4, axis: -1)
            let i = sigmoid(parts[0]), f = sigmoid(parts[1])
            let g = tanh(parts[2]), o = sigmoid(parts[3])
            c = f * c + i * g
            h = o * tanh(c)
            allH.append(h)
        }
        return stacked(allH, axis: -2)
    }

    private func backwardDir(_ x: MLXArray) -> MLXArray {
        let xProj = addMM(bias_ih_backward + bias_hh_backward, x, Wx_backward.transposed())
        let seqLen = x.dim(-2)
        var h = MLXArray.zeros([x.dim(0), hiddenSize])
        var c = MLXArray.zeros([x.dim(0), hiddenSize])
        var allH = [MLXArray]()
        for t in stride(from: seqLen - 1, through: 0, by: -1) {
            var ifgo = xProj[.ellipsis, t, 0...]
            ifgo = ifgo + h.matmul(Wh_backward.transposed())
            let parts = ifgo.split(parts: 4, axis: -1)
            let i = sigmoid(parts[0]), f = sigmoid(parts[1])
            let g = tanh(parts[2]), o = sigmoid(parts[3])
            c = f * c + i * g
            h = o * tanh(c)
            allH.insert(h, at: 0)
        }
        return stacked(allH, axis: -2)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, Any?) {
        var inp = x
        if inp.ndim == 2 { inp = inp.expandedDimensions(axis: 0) }
        let fwd = forwardDir(inp)
        let bwd = backwardDir(inp)
        return (concatenated([fwd, bwd], axis: -1), nil)
    }
}

// MARK: - ALBERT components

struct AlbertConfig {
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var embeddingSize: Int = 128
    var innerGroupNum: Int = 1
    var numHiddenGroups: Int = 1
    var hiddenDropoutProb: Float = 0.1
    var attentionProbsDropoutProb: Float = 0.1
    var typeVocabSize: Int = 2
    var layerNormEps: Float = 1e-12
    var vocabSize: Int = 30522
}

class AlbertEmbeddings: Module {
    let wordEmbeddings: Embedding
    let positionEmbeddings: Embedding
    let tokenTypeEmbeddings: Embedding
    let layerNorm: LayerNorm
    let drop: Dropout

    init(_ c: AlbertConfig) {
        wordEmbeddings = Embedding(embeddingCount: c.vocabSize, dimensions: c.embeddingSize)
        positionEmbeddings = Embedding(embeddingCount: c.maxPositionEmbeddings, dimensions: c.embeddingSize)
        tokenTypeEmbeddings = Embedding(embeddingCount: c.typeVocabSize, dimensions: c.embeddingSize)
        layerNorm = LayerNorm(dimensions: c.embeddingSize, eps: c.layerNormEps)
        drop = Dropout(p: c.hiddenDropoutProb)
    }

    func callAsFunction(_ inputIds: MLXArray, tokenTypeIds: MLXArray? = nil) -> MLXArray {
        let seqLen = inputIds.dim(1)
        let posIds = MLXArray(0..<Int32(seqLen)).expandedDimensions(axis: 0)
        let ttIds = tokenTypeIds ?? MLXArray.zeros(like: inputIds)
        var e = wordEmbeddings(inputIds) + positionEmbeddings(posIds) + tokenTypeEmbeddings(ttIds)
        e = layerNorm(e)
        e = drop(e)
        return e
    }
}

class AlbertSelfAttention: Module {
    let numHeads: Int
    let headSize: Int
    let allHeadSize: Int
    let query: Linear
    let key: Linear
    let value: Linear
    let dense: Linear
    let LayerNorm: LayerNorm
    let drop: Dropout

    init(_ c: AlbertConfig) {
        numHeads = c.numAttentionHeads
        headSize = c.hiddenSize / c.numAttentionHeads
        allHeadSize = numHeads * headSize
        query = Linear(c.hiddenSize, allHeadSize)
        key = Linear(c.hiddenSize, allHeadSize)
        value = Linear(c.hiddenSize, allHeadSize)
        dense = Linear(c.hiddenSize, c.hiddenSize)
        LayerNorm = MLXNN.LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        drop = Dropout(p: c.attentionProbsDropoutProb)
    }

    private func transposeForScores(_ x: MLXArray) -> MLXArray {
        let shape = x.shape.dropLast() + [numHeads, headSize]
        return x.reshaped(Array(shape)).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let q = transposeForScores(query(hidden))
        let k = transposeForScores(key(hidden))
        let v = transposeForScores(value(hidden))
        var scores = q.matmul(k.transposed(0, 1, 3, 2))
        scores = scores / Float(headSize).squareRoot()
        if let m = mask { scores = scores + m }
        var probs = softmax(scores, axis: -1)
        probs = drop(probs)
        var ctx = probs.matmul(v).transposed(0, 2, 1, 3)
        let outShape = ctx.shape.dropLast(2) + [allHeadSize]
        ctx = ctx.reshaped(Array(outShape))
        ctx = dense(ctx)
        ctx = LayerNorm(ctx + hidden)
        return ctx
    }
}

class AlbertLayer: Module {
    let attention: AlbertSelfAttention
    let full_layer_layer_norm: LayerNorm
    let ffn: Linear
    let ffn_output: Linear
    let activation = GELU()

    init(_ c: AlbertConfig) {
        attention = AlbertSelfAttention(c)
        full_layer_layer_norm = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        ffn = Linear(c.hiddenSize, c.intermediateSize)
        ffn_output = Linear(c.intermediateSize, c.hiddenSize)
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let attnOut = attention(hidden, mask: mask)
        var ff = ffn(attnOut)
        ff = activation(ff)
        ff = ffn_output(ff)
        return full_layer_layer_norm(ff + attnOut)
    }
}

class AlbertLayerGroup: Module {
    let albert_layers: [AlbertLayer]

    init(_ c: AlbertConfig) {
        albert_layers = (0..<c.innerGroupNum).map { _ in AlbertLayer(c) }
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = hidden
        for l in albert_layers { h = l(h, mask: mask) }
        return h
    }
}

class AlbertEncoder: Module {
    let config: AlbertConfig
    let embedding_hidden_mapping_in: Linear
    let albert_layer_groups: [AlbertLayerGroup]

    init(_ c: AlbertConfig) {
        config = c
        embedding_hidden_mapping_in = Linear(c.embeddingSize, c.hiddenSize)
        albert_layer_groups = (0..<c.numHiddenGroups).map { _ in AlbertLayerGroup(c) }
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = embedding_hidden_mapping_in(hidden)
        for i in 0..<config.numHiddenLayers {
            let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)
            h = albert_layer_groups[groupIdx](h, mask: mask)
        }
        return h
    }
}

class CustomAlbert: Module {
    let config: AlbertConfig
    let embeddings: AlbertEmbeddings
    let encoder: AlbertEncoder
    let pooler: Linear

    init(_ c: AlbertConfig) {
        config = c
        embeddings = AlbertEmbeddings(c)
        encoder = AlbertEncoder(c)
        pooler = Linear(c.hiddenSize, c.hiddenSize)
    }

    func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let emb = embeddings(inputIds)
        var mask: MLXArray? = nil
        if let am = attentionMask {
            mask = am.expandedDimensions(axes: [1, 2])
            mask = (1.0 - mask!) * (-10000.0)
        }
        let enc = encoder(emb, mask: mask)
        let pooled = tanh(pooler(enc[0..., 0, 0...]))
        return (enc, pooled)
    }
}

// MARK: - AdaLayerNorm

class AdaLayerNorm: Module {
    let channels: Int
    let eps: Float
    let fc: Linear

    init(styleDim: Int, channels: Int, eps: Float = 1e-5) {
        self.channels = channels
        self.eps = eps
        fc = Linear(styleDim, channels * 2)
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var h = fc(s)
        h = h.reshaped(h.dim(0), h.dim(1), 1)
        let parts = h.split(parts: 2, axis: 1)
        let gamma = parts[0].transposed(2, 0, 1)
        let beta = parts[1].transposed(2, 0, 1)
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let xn = (x - mean) / sqrt(variance + eps)
        return (1 + gamma) * xn + beta
    }
}

// MARK: - TextEncoder

class TextEncoder: Module {
    let embedding: Embedding
    let cnn: [[(any Module & Sendable)]]
    let lstm: BiLSTM

    init(channels: Int, kernelSize: Int, depth: Int, nSymbols: Int) {
        embedding = Embedding(embeddingCount: nSymbols, dimensions: channels)
        let padding = (kernelSize - 1) / 2
        var layers = [[(any Module & Sendable)]]()
        for _ in 0..<depth {
            layers.append([
                ConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize, padding: padding),
                LayerNorm(dimensions: channels),
                LeakyReLU(negativeSlope: 0.2),
                Dropout(p: 0.2),
            ])
        }
        cnn = layers
        lstm = BiLSTM(inputSize: channels, hiddenSize: channels / 2)
    }

    func callAsFunction(_ x: MLXArray, _ inputLengths: MLXArray, _ m: MLXArray) -> MLXArray {
        var out = embedding(x).transposed(0, 2, 1)
        let mask = m.expandedDimensions(axis: 1)
        out = which(mask, 0.0, out)
        for block in cnn {
            for layer in block {
                if let cw = layer as? ConvWeighted {
                    out = out.swappedAxes(2, 1)
                    out = cw.forward(out, conv: conv1d)
                    out = out.swappedAxes(2, 1)
                } else if let ln = layer as? LayerNorm {
                    out = out.swappedAxes(2, 1)
                    out = ln(out)
                    out = out.swappedAxes(2, 1)
                } else if let act = layer as? LeakyReLU {
                    out = act(out)
                } else if let d = layer as? Dropout {
                    out = d(out)
                }
                out = which(mask, 0.0, out)
            }
        }
        out = out.swappedAxes(2, 1)
        (out, _) = lstm(out)
        out = out.swappedAxes(2, 1)
        let xPad = MLXArray.zeros([out.dim(0), out.dim(1), mask.dim(-1)])
        // Copy available data
        var result = xPad
        // Use indexing to set values
        result[0..., 0..., ..<out.dim(2)] = out
        result = which(mask, 0.0, result)
        return result
    }
}

// MARK: - DurationEncoder

class DurationEncoder: Module {
    let lstms: [(any Module & Sendable)]
    let dModel: Int
    let styDim: Int

    init(styDim: Int, dModel: Int, nlayers: Int, dropout: Float = 0.1) {
        self.dModel = dModel
        self.styDim = styDim
        var blocks = [(any Module & Sendable)]()
        for _ in 0..<nlayers {
            blocks.append(BiLSTM(inputSize: dModel + styDim, hiddenSize: dModel / 2))
            blocks.append(AdaLayerNorm(styleDim: styDim, channels: dModel))
        }
        self.lstms = blocks
    }

    func callAsFunction(_ x: MLXArray, _ style: MLXArray, _ textLengths: MLXArray, _ m: MLXArray) -> MLXArray {
        var out = x.transposed(2, 0, 1)
        let s = broadcast(style, to: [out.dim(0), out.dim(1), style.dim(-1)])
        out = concatenated([out, s], axis: -1)
        out = which(m[.ellipsis, .newAxis].transposed(1, 0, 2), 0.0, out)
        out = out.transposed(1, 2, 0)

        for block in lstms {
            if let aln = block as? AdaLayerNorm {
                out = aln(out.transposed(0, 2, 1), style).transposed(0, 2, 1)
                out = concatenated([out, s.transposed(1, 2, 0)], axis: 1)
                out = which(m[.ellipsis, .newAxis].transposed(0, 2, 1), 0.0, out)
            } else if let bilstm = block as? BiLSTM {
                out = out.transposed(0, 2, 1)[0]
                (out, _) = bilstm(out)
                out = out.transposed(0, 2, 1)
                let xPad = MLXArray.zeros([out.dim(0), out.dim(1), m.dim(-1)])
                let padded = xPad
                padded[0..., 0..., ..<out.dim(2)] = out
                out = padded
            }
        }
        return out.transposed(0, 2, 1)
    }
}

// MARK: - ProsodyPredictor

class ProsodyPredictor: Module {
    let text_encoder: DurationEncoder
    let lstm: BiLSTM
    let duration_proj: LinearNorm
    let shared: BiLSTM
    let F0: [AdainResBlk1d]
    let N: [AdainResBlk1d]
    let F0_proj: Conv1d
    let N_proj: Conv1d

    // Provide accessors with original names for external code
    var durEncoder: DurationEncoder { text_encoder }
    var durationProj: LinearNorm { duration_proj }
    var f0Proj: Conv1d { F0_proj }
    var nProj: Conv1d { N_proj }

    init(styleDim: Int, dHid: Int, nlayers: Int, maxDur: Int = 50, dropout: Float = 0.1) {
        text_encoder = DurationEncoder(styDim: styleDim, dModel: dHid, nlayers: nlayers, dropout: dropout)
        lstm = BiLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)
        duration_proj = LinearNorm(inDim: dHid, outDim: maxDur)
        shared = BiLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)
        F0 = [
            AdainResBlk1d(dimIn: dHid, dimOut: dHid, styleDim: styleDim, dropoutP: dropout, useConv1d: true),
            AdainResBlk1d(dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: true, dropoutP: dropout, useConv1d: true),
            AdainResBlk1d(dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim, dropoutP: dropout, useConv1d: true),
        ]
        N = [
            AdainResBlk1d(dimIn: dHid, dimOut: dHid, styleDim: styleDim, dropoutP: dropout, useConv1d: true),
            AdainResBlk1d(dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: true, dropoutP: dropout, useConv1d: true),
            AdainResBlk1d(dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim, dropoutP: dropout, useConv1d: true),
        ]
        F0_proj = Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0)
        N_proj = Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0)
    }

    func f0nTrain(_ x: MLXArray, _ s: MLXArray) -> (MLXArray, MLXArray) {
        var shared_out: MLXArray
        (shared_out, _) = shared(x.transposed(0, 2, 1))

        var f0 = shared_out.transposed(0, 2, 1)
        for block in F0 { f0 = block(f0, s) }
        f0 = f0Proj(f0.swappedAxes(2, 1)).swappedAxes(2, 1)

        var n = shared_out.transposed(0, 2, 1)
        for block in N { n = block(n, s) }
        n = nProj(n.swappedAxes(2, 1)).swappedAxes(2, 1)

        return (f0.squeezed(axis: 1), n.squeezed(axis: 1))
    }
}
