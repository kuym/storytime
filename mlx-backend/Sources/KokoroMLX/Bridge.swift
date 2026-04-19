// C-callable entry points for the Kokoro MLX backend.
// These are linked from the Rust CLI via FFI.

import Foundation
import MLX
import MLXNN

// Global model state
private var globalModel: KokoroModel? = nil

/// Initialize the Kokoro model from a directory containing config.json and *.safetensors.
/// Returns 0 on success, -1 on failure.
@_cdecl("kokoro_init")
public func kokoroInit(weightsDir: UnsafePointer<CChar>) -> Int32 {
    let dir = String(cString: weightsDir)
    do {
        let configURL = URL(fileURLWithPath: dir).appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let json = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        let config = parseConfig(json)

        let model = KokoroModel(config)

        // Load safetensors weights
        let dirURL = URL(fileURLWithPath: dir)
        let files = try FileManager.default.contentsOfDirectory(at: dirURL,
            includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }

        var allWeights = [String: MLXArray]()
        for file in files {
            let w = try loadArrays(url: file)
            for (k, v) in w { allWeights[k] = v }
        }

        let sanitized = sanitizeWeights(allWeights)
        let params = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: params, verify: .noUnusedKeys)
        eval(model)

        globalModel = model
        return 0
    } catch {
        fputs("kokoro_init error: \(error)\n", stderr)
        return -1
    }
}

/// Generate audio from IPA phonemes and a voice file.
/// `phonemes`: null-terminated IPA string
/// `voicePath`: path to a .safetensors voice file
/// `speed`: speaking rate multiplier
/// `nTokens`: number of phoneme tokens (used for style row selection)
/// `audioOut`: receives a pointer to float32 audio samples (caller must free with kokoro_free_audio)
/// `audioLen`: receives the number of samples
/// Returns 0 on success, -1 on failure.
@_cdecl("kokoro_generate")
public func kokoroGenerate(
    phonemes: UnsafePointer<CChar>,
    voicePath: UnsafePointer<CChar>,
    speed: Float,
    nTokens: Int32,
    audioOut: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>,
    audioLen: UnsafeMutablePointer<Int32>
) -> Int32 {
    guard let model = globalModel else {
        fputs("kokoro_generate: model not initialized\n", stderr)
        return -1
    }
    do {
        let ipa = String(cString: phonemes)
        let vPath = String(cString: voicePath)

        // Load voice
        let voiceArrays = try loadArrays(url: URL(fileURLWithPath: vPath))
        guard let voiceTensor = voiceArrays["voice"] ?? voiceArrays.values.first else {
            fputs("kokoro_generate: no voice tensor in \(vPath)\n", stderr)
            return -1
        }

        // Select style row based on number of tokens
        let idx = min(Int(nTokens), voiceTensor.dim(0) - 1)
        let refS = voiceTensor[idx].expandedDimensions(axis: 0)

        let audio = model.forward(phonemes: ipa, refS: refS, speed: speed)
        eval(audio)

        let floats = audio.asType(.float32).asArray(Float.self)
        let count = floats.count
        let buf = UnsafeMutablePointer<Float>.allocate(capacity: count)
        floats.withUnsafeBufferPointer { src in
            buf.initialize(from: src.baseAddress!, count: count)
        }
        audioOut.pointee = buf
        audioLen.pointee = Int32(count)
        return 0
    } catch {
        fputs("kokoro_generate error: \(error)\n", stderr)
        return -1
    }
}

/// Free audio buffer allocated by kokoro_generate.
@_cdecl("kokoro_free_audio")
public func kokoroFreeAudio(audio: UnsafeMutablePointer<Float>?) {
    audio?.deallocate()
}

/// Release the model.
@_cdecl("kokoro_cleanup")
public func kokoroCleanup() {
    globalModel = nil
}

/// Get the sample rate.
@_cdecl("kokoro_sample_rate")
public func kokoroSampleRate() -> Int32 {
    return Int32(globalModel?.config.sampleRate ?? 24000)
}
