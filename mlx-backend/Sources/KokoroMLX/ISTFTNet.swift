// iSTFT-based vocoder for Kokoro. Ported from mlx-audio's istftnet.py.

import Foundation
import MLX
import MLXFFT
import MLXNN
import Numerics

// MARK: - Weight normalization

func computeNorm(_ x: MLXArray, p: Int, dims: [Int], keepDims: Bool = false) -> MLXArray {
    if p == 1 {
        return abs(x).sum(axes: dims, keepDims: keepDims)
    }
    return sqrt((x * x).sum(axes: dims, keepDims: keepDims))
}

func weightNorm(_ v: MLXArray, _ g: MLXArray, dim: Int = 0) -> MLXArray {
    let rank = v.ndim
    var axes = Array(0..<rank)
    let d = dim < 0 ? dim + rank : dim
    if d != -1 && d < rank { axes.removeAll { $0 == d } }
    let normV = computeNorm(v, p: 2, dims: axes, keepDims: true) + 1e-7
    return (v / normV) * g
}

// MARK: - ConvWeighted

class ConvWeighted: Module {
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1,
         padding: Int = 1, dilation: Int = 1, groups: Int = 1,
         bias: Bool = true, encode: Bool = false) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        weight_g = MLXArray.ones([outChannels, 1, 1])
        weight_v = MLXArray.ones([outChannels, kernelSize, inChannels])
        self.bias = bias ? MLXArray.zeros([encode ? inChannels : outChannels]) : nil
    }

    func forward(_ x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
        let w = weightNorm(weight_v, weight_g, dim: 0)
        let b = bias?.reshaped(1, 1, -1)
        let useW: MLXArray
        if x.dim(-1) == w.dim(-1) || groups > 1 {
            useW = w
        } else {
            useW = w.transposed()
        }
        var out = conv(x, useW, stride, padding, dilation, groups, .default)
        if let b = b { out = out + b }
        return out
    }

    // Overload for convTransposed1d which has an extra outputPadding parameter
    func forwardTransposed(_ x: MLXArray) -> MLXArray {
        let w = weightNorm(weight_v, weight_g, dim: 0)
        let b = bias?.reshaped(1, 1, -1)
        let useW: MLXArray
        if x.dim(-1) == w.dim(-1) || groups > 1 {
            useW = w
        } else {
            useW = w.transposed()
        }
        var out = convTransposed1d(x, useW, stride: stride, padding: padding,
                                    dilation: dilation, outputPadding: 0, groups: groups)
        if let b = b { out = out + b }
        return out
    }
}

// MARK: - InstanceNorm1d

class InstanceNorm1d: Module {
    let eps: Float

    init(numFeatures: Int, eps: Float = 1e-5) {
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: (N, C, L) — normalize over L dimension
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        return (x - mean) / sqrt(variance + eps)
    }
}

// MARK: - AdaIN1d

class AdaIN1d: Module {
    let norm: InstanceNorm1d
    let fc: Linear

    init(styleDim: Int, numFeatures: Int) {
        norm = InstanceNorm1d(numFeatures: numFeatures)
        fc = Linear(styleDim, numFeatures * 2)
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let h = fc(s).expandedDimensions(axis: 2)
        let parts = h.split(parts: 2, axis: 1)
        return (1 + parts[0]) * norm(x) + parts[1]
    }
}

// MARK: - AdaINResBlock1 (for Generator)

class AdaINResBlock1: Module {
    let convs1: [ConvWeighted]
    let convs2: [ConvWeighted]
    let adain1: [AdaIN1d]
    let adain2: [AdaIN1d]
    var alpha1: [MLXArray]
    var alpha2: [MLXArray]

    init(channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5], styleDim: Int = 64) {
        convs1 = dilation.map { d in
            ConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                         stride: 1, padding: (kernelSize * d - d) / 2, dilation: d)
        }
        convs2 = (0..<3).map { _ in
            ConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                         stride: 1, padding: (kernelSize - 1) / 2, dilation: 1)
        }
        adain1 = (0..<3).map { _ in AdaIN1d(styleDim: styleDim, numFeatures: channels) }
        adain2 = (0..<3).map { _ in AdaIN1d(styleDim: styleDim, numFeatures: channels) }
        alpha1 = (0..<3).map { _ in MLXArray.ones([1, channels, 1]) }
        alpha2 = (0..<3).map { _ in MLXArray.ones([1, channels, 1]) }
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var out = x
        for (c1, c2, n1, n2, a1, a2) in zip6(convs1, convs2, adain1, adain2, alpha1, alpha2) {
            var xt = n1(out, s)
            xt = xt + (1 / a1) * pow(sin(a1 * xt), 2)
            xt = c1.forward(xt.swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * pow(sin(a2 * xt), 2)
            xt = c2.forward(xt.swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
            out = xt + out
        }
        return out
    }
}

func zip6<A, B, C, D, E, F>(_ a: [A], _ b: [B], _ c: [C], _ d: [D], _ e: [E], _ f: [F])
    -> [(A, B, C, D, E, F)] {
    (0..<min(a.count, b.count, c.count, d.count, e.count, f.count)).map {
        (a[$0], b[$0], c[$0], d[$0], e[$0], f[$0])
    }
}

// MARK: - AdainResBlk1d (for Decoder / ProsodyPredictor)

class AdainResBlk1d: Module {
    let conv1: ConvWeighted
    let conv2: ConvWeighted
    let norm1: AdaIN1d
    let norm2: AdaIN1d
    let leakyRelu = LeakyReLU(negativeSlope: 0.2)
    let dropout: Dropout
    let learnedSc: Bool
    var conv1x1: ConvWeighted?
    let upsampleType: String
    let upsampleLayer: Upsample?
    let pool: ConvWeighted?
    let useConv1d: Bool

    init(dimIn: Int, dimOut: Int, styleDim: Int = 64, upsample: Bool = false,
         dropoutP: Float = 0.0, useConv1d: Bool = false) {
        self.useConv1d = useConv1d
        upsampleType = upsample ? "timepreserve" : "none"
        upsampleLayer = upsample ? Upsample(scaleFactor: .float(2.0), mode: .nearest) : nil
        learnedSc = dimIn != dimOut
        conv1 = ConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 3, stride: 1, padding: 1)
        conv2 = ConvWeighted(inChannels: dimOut, outChannels: dimOut, kernelSize: 3, stride: 1, padding: 1)
        norm1 = AdaIN1d(styleDim: styleDim, numFeatures: dimIn)
        norm2 = AdaIN1d(styleDim: styleDim, numFeatures: dimOut)
        dropout = Dropout(p: dropoutP)
        if learnedSc {
            conv1x1 = ConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 1, stride: 1, padding: 0, bias: false)
        }
        pool = upsample ? ConvWeighted(inChannels: 1, outChannels: dimIn, kernelSize: 3, stride: 2, padding: 1, groups: dimIn) : nil
    }

    private func shortcut(_ x: MLXArray) -> MLXArray {
        var out = x
        if let up = upsampleLayer {
            out = up(out.swappedAxes(2, 1)).swappedAxes(2, 1)
        }
        if let c = conv1x1 {
            out = c.forward(out.swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
        }
        return out
    }

    private func residual(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var out = leakyRelu(norm1(x, s))
        if let p = pool, upsampleType != "none" {
            out = p.forwardTransposed(out.swappedAxes(2, 1)).swappedAxes(2, 1)
            out = padded(out, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0))])
        }
        out = conv1.forward(dropout(out.swappedAxes(2, 1)), conv: conv1d).swappedAxes(2, 1)
        out = leakyRelu(norm2(out, s))
        out = conv2.forward(out.swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
        return out
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        (residual(x, s) + shortcut(x)) / sqrt(MLXArray(2.0))
    }
}

// MARK: - STFT / iSTFT

func hannWindow(_ n: Int) -> MLXArray {
    let indices = MLXArray(0..<Int32(n))
    return 0.5 * (1.0 - cos(2.0 * Float.pi * indices / Float(n)))
}

func stftTransform(_ x: MLXArray, nFFT: Int, hopLength: Int) -> (MLXArray, MLXArray) {
    let w = hannWindow(nFFT)
    let padLen = nFFT / 2
    // Reverse using slice with negative stride isn't available; use array indexing
    let prefixIndices = MLXArray(Array(stride(from: Int32(padLen), through: 1, by: -1)))
    let suffixIndices = MLXArray(Array(stride(from: Int32(x.dim(0) - 2), through: Int32(x.dim(0) - padLen - 1), by: -1)))
    let prefix = x[prefixIndices]
    let suffix = x[suffixIndices]
    let padX = concatenated([prefix, x, suffix])
    let numFrames = 1 + (padX.dim(0) - nFFT) / hopLength
    let frames = asStrided(padX, [numFrames, nFFT], strides: [hopLength, 1])
    let windowed = frames * w
    let spec = rfft(windowed)
    let magnitude = abs(spec)
    let phase = atan2(spec.imaginaryPart(), spec.realPart())
    return (magnitude, phase)
}

func istftInverse(_ mag: MLXArray, _ phase: MLXArray, hopLength: Int, winLength: Int) -> MLXArray {
    let w = hannWindow(winLength + 1)[0..<winLength]
    let numFrames = mag.dim(0)
    let real = mag * cos(phase)
    let imag = mag * sin(phase)
    let spec = real.asType(.complex64) + imag.asType(.complex64) * MLXArray(Complex<Float>(0, 1))
    let framesTime = MLXFFT.irfft(spec).transposed(1, 0)
    let t = (numFrames - 1) * hopLength + winLength
    var reconstructed = MLXArray.zeros([t])
    var windowSum = MLXArray.zeros([t])
    let offsets = MLXArray(0..<Int32(numFrames)) * Int32(hopLength)
    let indices = offsets.expandedDimensions(axis: 1) + MLXArray(0..<Int32(winLength))
    let flat = indices.flattened()
    let updatesR = (framesTime * w).flattened()
    let wBroadcast = broadcast(w, to: [numFrames, winLength]).flattened()
    reconstructed = reconstructed.at[flat].add(updatesR)
    windowSum = windowSum.at[flat].add(wBroadcast)
    reconstructed = which(windowSum .> 1e-10, reconstructed / windowSum, reconstructed)
    let start = winLength / 2
    let end = t - winLength / 2
    return reconstructed[start..<end]
}

// MARK: - SineGen + SourceModule

class SourceModuleHnNSF: Module {
    let sineAmp: Float
    let noiseStd: Float
    let samplingRate: Int
    let harmonicNum: Int
    let upsampleScale: Int
    let voicedThreshold: Float
    let lLinear: Linear

    init(samplingRate: Int, upsampleScale: Int, harmonicNum: Int = 0,
         sineAmp: Float = 0.1, addNoiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        self.sineAmp = sineAmp
        self.noiseStd = addNoiseStd
        self.samplingRate = samplingRate
        self.harmonicNum = harmonicNum
        self.upsampleScale = upsampleScale
        self.voicedThreshold = voicedThreshold
        lLinear = Linear(harmonicNum + 1, 1)
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let fn = f0 * MLXArray(1...Int32(harmonicNum + 1)).expandedDimensions(axes: [0, 1])
        let radValues = remainder(fn / Float(samplingRate), 1.0)
        let phase = cumsum(radValues, axis: 1) * (2.0 * Float.pi)
        let sines = sin(phase) * sineAmp
        let uv = (f0 .> voicedThreshold).asType(.float32)
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
        let noise = noiseAmp * MLXRandom.normal(sines.shape)
        let sineWavs = sines * uv + noise
        let sineMerge = tanh(lLinear(sineWavs))
        let noiseOut = MLXRandom.normal(uv.shape) * sineAmp / 3
        return (sineMerge, noiseOut, uv)
    }
}

// MARK: - Generator

class Generator: Module {
    let numKernels: Int
    let numUpsamples: Int
    let m_source: SourceModuleHnNSF
    let f0_upsamp: Upsample
    let noise_convs: [Conv1d]
    let noise_res: [AdaINResBlock1]
    let ups: [ConvWeighted]
    let resblocks: [AdaINResBlock1]
    let postNFFT: Int
    let conv_post: ConvWeighted
    let stftNFFT: Int
    let stftHopSize: Int

    // Accessors for external code using camelCase
    var mSource: SourceModuleHnNSF { m_source }
    var f0Upsamp: Upsample { f0_upsamp }
    var noiseConvs: [Conv1d] { noise_convs }
    var noiseRes: [AdaINResBlock1] { noise_res }
    var convPost: ConvWeighted { conv_post }

    init(styleDim: Int, resblockKernelSizes: [Int], upsampleRates: [Int],
         upsampleInitialChannel: Int, resblockDilationSizes: [[Int]],
         upsampleKernelSizes: [Int], genIstftNFFT: Int, genIstftHopSize: Int) {
        numKernels = resblockKernelSizes.count
        numUpsamples = upsampleRates.count
        stftNFFT = genIstftNFFT
        stftHopSize = genIstftHopSize
        let totalUpsample = upsampleRates.reduce(1, *) * genIstftHopSize
        m_source = SourceModuleHnNSF(samplingRate: 24000, upsampleScale: totalUpsample, harmonicNum: 8, voicedThreshold: 10)
        f0_upsamp = Upsample(scaleFactor: .float(Float(totalUpsample)))

        var upsArr = [ConvWeighted]()
        var noiseC = [Conv1d]()
        var noiseR = [AdaINResBlock1]()
        var resb = [AdaINResBlock1]()

        for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
            upsArr.append(ConvWeighted(
                inChannels: upsampleInitialChannel / (1 << (i + 1)),
                outChannels: upsampleInitialChannel / (1 << i),
                kernelSize: k, stride: u, padding: (k - u) / 2, encode: true
            ))
            let cCur = upsampleInitialChannel / (1 << (i + 1))
            if i + 1 < upsampleRates.count {
                let strideF0 = upsampleRates[(i+1)...].reduce(1, *)
                noiseC.append(Conv1d(inputChannels: genIstftNFFT + 2, outputChannels: cCur,
                                     kernelSize: strideF0 * 2, stride: strideF0, padding: (strideF0 + 1) / 2))
                noiseR.append(AdaINResBlock1(channels: cCur, kernelSize: 7, dilation: [1, 3, 5], styleDim: styleDim))
            } else {
                noiseC.append(Conv1d(inputChannels: genIstftNFFT + 2, outputChannels: cCur, kernelSize: 1))
                noiseR.append(AdaINResBlock1(channels: cCur, kernelSize: 11, dilation: [1, 3, 5], styleDim: styleDim))
            }
            for (k2, d) in zip(resblockKernelSizes, resblockDilationSizes) {
                resb.append(AdaINResBlock1(channels: cCur, kernelSize: k2, dilation: d, styleDim: styleDim))
            }
        }
        ups = upsArr
        noise_convs = noiseC
        noise_res = noiseR
        resblocks = resb
        postNFFT = genIstftNFFT
        conv_post = ConvWeighted(inChannels: upsampleInitialChannel / (1 << numUpsamples),
                                outChannels: genIstftNFFT + 2, kernelSize: 7, stride: 1, padding: 3)
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ f0: MLXArray) -> MLXArray {
        let f0Up = f0Upsamp(f0.expandedDimensions(axis: 1).transposed(0, 2, 1))
        let (harSource, _, _) = mSource(f0Up)
        let harSourceSq = harSource.transposed(0, 2, 1).squeezed(axis: 1)
        let (harSpec, harPhase) = stftTransform(harSourceSq[0], nFFT: stftNFFT, hopLength: stftHopSize)
        var har = concatenated([harSpec, harPhase], axis: 1).expandedDimensions(axis: 0).swappedAxes(2, 1)
        var out = x
        for i in 0..<numUpsamples {
            out = leakyReLU(out, negativeSlope: 0.1)
            var xSource = noiseConvs[i](har).swappedAxes(2, 1)
            xSource = noiseRes[i](xSource, s)
            out = ups[i].forwardTransposed(out.swappedAxes(2, 1)).swappedAxes(2, 1)
            if i == numUpsamples - 1 {
                out = padded(out, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0))])
            }
            out = out + xSource
            var xs: MLXArray? = nil
            for j in 0..<numKernels {
                let r = resblocks[i * numKernels + j](out, s)
                xs = xs.map { $0 + r } ?? r
            }
            out = xs! / Float(numKernels)
        }
        out = leakyReLU(out, negativeSlope: 0.01)
        out = convPost.forward(out.swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
        let spec = exp(out[0..., ..<(postNFFT / 2 + 1), 0...])
        let phase = sin(out[0..., (postNFFT / 2 + 1)..., 0...])
        let audio = istftInverse(spec[0].transposed(), phase[0].transposed(), hopLength: stftHopSize, winLength: stftNFFT)
        return audio.expandedDimensions(axes: [0, 1])
    }
}

// MARK: - Decoder

class Decoder: Module {
    let encode: AdainResBlk1d
    let decode: [AdainResBlk1d]
    let F0_conv: ConvWeighted
    let N_conv: ConvWeighted
    let asr_res: [ConvWeighted]
    let generator: Generator

    // Accessors
    var f0Conv: ConvWeighted { F0_conv }
    var nConv: ConvWeighted { N_conv }
    var asrRes: [ConvWeighted] { asr_res }

    init(dimIn: Int, styleDim: Int, dimOut: Int,
         resblockKernelSizes: [Int], upsampleRates: [Int],
         upsampleInitialChannel: Int, resblockDilationSizes: [[Int]],
         upsampleKernelSizes: [Int], genIstftNFFT: Int, genIstftHopSize: Int) {
        encode = AdainResBlk1d(dimIn: dimIn + 2, dimOut: 1024, styleDim: styleDim, useConv1d: true)
        decode = [
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim, useConv1d: true),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim, useConv1d: true),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim, useConv1d: true),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: styleDim, upsample: true, useConv1d: true),
        ]
        F0_conv = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1)
        N_conv = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1)
        asr_res = [ConvWeighted(inChannels: 512, outChannels: 64, kernelSize: 1, padding: 0)]
        generator = Generator(styleDim: styleDim, resblockKernelSizes: resblockKernelSizes,
                              upsampleRates: upsampleRates, upsampleInitialChannel: upsampleInitialChannel,
                              resblockDilationSizes: resblockDilationSizes,
                              upsampleKernelSizes: upsampleKernelSizes,
                              genIstftNFFT: genIstftNFFT, genIstftHopSize: genIstftHopSize)
    }

    func callAsFunction(_ asr: MLXArray, _ f0Curve: MLXArray, _ n: MLXArray, _ s: MLXArray) -> MLXArray {
        let f0 = f0Conv.forward(f0Curve.expandedDimensions(axis: 1).swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
        let nDown = nConv.forward(n.expandedDimensions(axis: 1).swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
        var x = concatenated([asr, f0, nDown], axis: 1)
        x = encode(x, s)
        let asrResOut = asrRes[0].forward(asr.swappedAxes(2, 1), conv: conv1d).swappedAxes(2, 1)
        var res = true
        for block in decode {
            if res { x = concatenated([x, asrResOut, f0, nDown], axis: 1) }
            x = block(x, s)
            if block.upsampleType != "none" { res = false }
        }
        return generator(x, s, f0Curve)
    }
}

// Helper for leaky relu
func leakyReLU(_ x: MLXArray, negativeSlope: Float) -> MLXArray {
    which(x .> 0, x, x * negativeSlope)
}
