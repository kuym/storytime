// Kokoro-82M top-level model. Ported from mlx-audio's kokoro.py.

import Foundation
import MLX
import MLXNN

// MARK: - Config

struct KokoroConfig {
    let istftnet: ISTFTNetConfig
    let dimIn: Int
    let dropout: Float
    let hiddenDim: Int
    let maxConvDim: Int
    let maxDur: Int
    let multispeaker: Bool
    let nLayer: Int
    let nMels: Int
    let nToken: Int
    let styleDim: Int
    let textEncoderKernelSize: Int
    let plbert: AlbertConfig
    let vocab: [String: Int]
    let sampleRate: Int

    struct ISTFTNetConfig {
        let upsampleKernelSizes: [Int]
        let upsampleRates: [Int]
        let genIstftHopSize: Int
        let genIstftNFFT: Int
        let resblockDilationSizes: [[Int]]
        let resblockKernelSizes: [Int]
        let upsampleInitialChannel: Int
    }
}

func parseConfig(_ json: [String: Any]) -> KokoroConfig {
    let ist = json["istftnet"] as! [String: Any]
    let plb = json["plbert"] as! [String: Any]
    let vocabRaw = json["vocab"] as! [String: Int]

    return KokoroConfig(
        istftnet: .init(
            upsampleKernelSizes: ist["upsample_kernel_sizes"] as! [Int],
            upsampleRates: ist["upsample_rates"] as! [Int],
            genIstftHopSize: ist["gen_istft_hop_size"] as! Int,
            genIstftNFFT: ist["gen_istft_n_fft"] as! Int,
            resblockDilationSizes: ist["resblock_dilation_sizes"] as! [[Int]],
            resblockKernelSizes: ist["resblock_kernel_sizes"] as! Int == 0 ? [] : ist["resblock_kernel_sizes"] as! [Int],
            upsampleInitialChannel: ist["upsample_initial_channel"] as! Int
        ),
        dimIn: json["dim_in"] as! Int,
        dropout: Float(json["dropout"] as! Double),
        hiddenDim: json["hidden_dim"] as! Int,
        maxConvDim: json["max_conv_dim"] as! Int,
        maxDur: json["max_dur"] as! Int,
        multispeaker: json["multispeaker"] as! Bool,
        nLayer: json["n_layer"] as! Int,
        nMels: json["n_mels"] as! Int,
        nToken: json["n_token"] as! Int,
        styleDim: json["style_dim"] as! Int,
        textEncoderKernelSize: json["text_encoder_kernel_size"] as! Int,
        plbert: AlbertConfig(
            numHiddenLayers: plb["num_hidden_layers"] as! Int,
            numAttentionHeads: plb["num_attention_heads"] as! Int,
            hiddenSize: plb["hidden_size"] as! Int,
            intermediateSize: plb["intermediate_size"] as! Int,
            maxPositionEmbeddings: plb["max_position_embeddings"] as! Int,
            hiddenDropoutProb: Float(plb["dropout"] as? Double ?? 0.1),
            attentionProbsDropoutProb: Float(plb["dropout"] as? Double ?? 0.1)
        ),
        vocab: vocabRaw,
        sampleRate: json["sample_rate"] as? Int ?? 24000
    )
}

// MARK: - Model

class KokoroModel: Module {
    let config: KokoroConfig
    let vocab: [String: Int]
    let bert: CustomAlbert
    let bert_encoder: Linear
    let predictor: ProsodyPredictor
    let text_encoder: TextEncoder
    let decoder: Decoder

    // Accessors for external code using camelCase
    var bertEncoder: Linear { bert_encoder }
    var textEncoder: TextEncoder { text_encoder }

    init(_ config: KokoroConfig) {
        self.config = config
        self.vocab = config.vocab
        var ac = config.plbert
        ac.vocabSize = config.nToken
        bert = CustomAlbert(ac)
        bert_encoder = Linear(config.plbert.hiddenSize, config.hiddenDim)
        predictor = ProsodyPredictor(styleDim: config.styleDim, dHid: config.hiddenDim,
                                     nlayers: config.nLayer, maxDur: config.maxDur,
                                     dropout: config.dropout)
        text_encoder = TextEncoder(channels: config.hiddenDim,
                                  kernelSize: config.textEncoderKernelSize,
                                  depth: config.nLayer, nSymbols: config.nToken)
        decoder = Decoder(dimIn: config.hiddenDim, styleDim: config.styleDim,
                          dimOut: config.nMels,
                          resblockKernelSizes: config.istftnet.resblockKernelSizes,
                          upsampleRates: config.istftnet.upsampleRates,
                          upsampleInitialChannel: config.istftnet.upsampleInitialChannel,
                          resblockDilationSizes: config.istftnet.resblockDilationSizes,
                          upsampleKernelSizes: config.istftnet.upsampleKernelSizes,
                          genIstftNFFT: config.istftnet.genIstftNFFT,
                          genIstftHopSize: config.istftnet.genIstftHopSize)
    }

    func forward(phonemes: String, refS: MLXArray, speed: Float = 1.0) -> MLXArray {
        let inputIds = phonemes.compactMap { vocab[String($0)] }
        let ids = MLXArray([Int32(0)] + inputIds.map { Int32($0) } + [Int32(0)]).expandedDimensions(axis: 0)
        let inputLengths = MLXArray([Int32(ids.dim(1))])
        var textMask = MLXArray(0..<Int32(ids.dim(1))).expandedDimensions(axis: 0)
        textMask = (textMask + 1) .> inputLengths.expandedDimensions(axis: 1)

        let (bertDur, _) = bert(ids, attentionMask: (.!textMask).asType(.int32))
        let dEn = bertEncoder(bertDur).transposed(0, 2, 1)

        let s = refS[0..., 128...]
        let d = predictor.durEncoder(dEn, s, inputLengths, textMask)
        var lstmOut: MLXArray
        (lstmOut, _) = predictor.lstm(d)
        var duration = predictor.durationProj(lstmOut)
        duration = sigmoid(duration).sum(axis: -1) / speed
        duration = clip(round(duration), min: 1, max: 100).asType(.int32)[0]

        // Build alignment
        var indices = [MLXArray]()
        for i in 0..<duration.dim(0) {
            let count = min(max(Int(duration[i].item(Int32.self)), 0), 100)
            if count > 0 {
                indices.append(full([count], values: Int32(i)))
            }
        }
        guard !indices.isEmpty else { return MLXArray.zeros([1]) }
        let idxConcat = concatenated(indices)
        var predAlnTrg = MLXArray.zeros([ids.dim(1), idxConcat.dim(0)])
        predAlnTrg[idxConcat, MLXArray(0..<Int32(idxConcat.dim(0)))] = MLXArray(Float(1.0))
        let predAln = predAlnTrg.expandedDimensions(axis: 0)

        let en = d.transposed(0, 2, 1).matmul(predAln)
        let (f0Pred, nPred) = predictor.f0nTrain(en, s)

        let tEn = textEncoder(ids, inputLengths, textMask)
        let asr = tEn.matmul(predAln)

        let audio = decoder(asr, f0Pred, nPred, refS[0..., ..<128])
        eval(audio)
        return audio[0, 0]
    }
}

// MARK: - Weight sanitization

func sanitizeLSTMKey(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
    let map: [String: String] = [
        "weight_ih_l0_reverse": "Wx_backward",
        "weight_hh_l0_reverse": "Wh_backward",
        "bias_ih_l0_reverse": "bias_ih_backward",
        "bias_hh_l0_reverse": "bias_hh_backward",
        "weight_ih_l0": "Wx_forward",
        "weight_hh_l0": "Wh_forward",
        "bias_ih_l0": "bias_ih_forward",
        "bias_hh_l0": "bias_hh_forward",
    ]
    let base = key.components(separatedBy: ".").dropLast().joined(separator: ".")
    for (suffix, newSuffix) in map {
        if key.hasSuffix(suffix) {
            return [("\(base).\(newSuffix)", value)]
        }
    }
    return [(key, value)]
}

func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var out = [String: MLXArray]()
    for (key, value) in weights {
        if key.contains("position_ids") { continue }

        if key.hasSuffix(".gamma") || key.hasSuffix(".beta") {
            let base = key.components(separatedBy: ".").dropLast().joined(separator: ".")
            let newKey = key.hasSuffix(".gamma") ? "\(base).weight" : "\(base).bias"
            out[newKey] = value
        } else if key.contains("weight_ih_l0") || key.contains("weight_hh_l0") ||
                    key.contains("bias_ih_l0") || key.contains("bias_hh_l0") {
            for (k, v) in sanitizeLSTMKey(key, value) { out[k] = v }
        } else if key.contains("weight_v") && value.ndim == 3 {
            // Check if already in MLX layout (out, kernel, in)
            if value.dim(1) <= value.dim(2) {
                out[key] = value
            } else {
                out[key] = value.transposed(0, 2, 1)
            }
        } else if key.contains("F0_proj.weight") || key.contains("N_proj.weight") {
            if value.ndim == 3 { out[key] = value.transposed(0, 2, 1) }
            else { out[key] = value }
        } else if key.contains("noise_convs") && key.hasSuffix(".weight") {
            if value.ndim == 3 { out[key] = value.transposed(0, 2, 1) }
            else { out[key] = value }
        } else {
            out[key] = value
        }
    }
    return out
}
