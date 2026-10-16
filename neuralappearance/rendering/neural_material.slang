// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
module neural_material_<HASH>;

__exported import falcor2.render;
import falcor2.utils;
import model.neural_material_model;
import rendering.neural_material_globals;
import util.texture_helpers;

// Runtime material instance for one shading point. It looks up the spatial
// latent code, then delegates BSDF evaluation and sampling to the trained
// decoder and sampler stored in the generated config type.
public struct NeuralMaterialInstance_<HASH> : IMaterialInstance
{
    int material_id = 0;
    int mip_level = 0;
    float texel_sample = 0.f;
    uint64_t config_handle = 0xFFFFFFFFFFFFFFFFull;
    // Decoded once per shading-point visit by setup_material_instance()
    // below and reused by eval/sample/eval_pdf, instead of each of them
    // independently re-running the latent fetch (a cheap buffer read for
    // the dense path, but a full grid-sample + decode-MLP forward pass
    // for NtcLatentTexture -- see rtcnam/results/README.md's "Render-time
    // cost" section). Safe because material_id/mip_level/texel_sample/
    // si.uv are already identical across all three calls for one hit.
    //
    // Only populated when `NeuralMaterialConfig_<HASH>.cache_latents` is
    // true, i.e. for NtcLatentTexture. The dense path leaves it at its
    // default and re-fetches per call through `get_latents()` below --
    // caching measured ~30% SLOWER there, because this is filled eagerly
    // before the `wi.z <= 0` early-outs and a dense fetch is cheaper than
    // the extra always-on work. See rendering/neural_material.py.
    float[<NUM_LATENTS>] cached_latents;

    __init(
        int material_id, int mip_level, float texel_sample, uint64_t config_handle,
        float[<NUM_LATENTS>] cached_latents
    )
    {
        this.material_id = material_id;
        this.mip_level = mip_level;
        this.texel_sample = texel_sample;
        this.config_handle = config_handle;
        this.cached_latents = cached_latents;
    }

    NeuralMaterialConfig_<HASH> get_config()
    {
        BufferHandle handle = { config_handle };
        return handle.load_aligned<NeuralMaterialConfig_<HASH>, 16>(0);
    }

    // Either hands back the value setup_material_instance() already decoded,
    // or does the fetch now -- a compile-time branch on a `static const`, so
    // exactly one of the two survives codegen for any given architecture.
    float[<NUM_LATENTS>] get_latents(const NeuralMaterialConfig_<HASH> config, const SurfaceInteraction si)
    {
        if (NeuralMaterialConfig_<HASH>.cache_latents)
            return this.cached_latents;
        return config.latents.get_bilinear_stochastic({ material_id, mip_level, si.uv }, texel_sample);
    }

    public float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo_ws, inout S sg)
    {
        float3 wi = si.shading_frame_ws.to_local(si.wi_ws);
        float3 wo = si.shading_frame_ws.to_local(wo_ws);
        if (wi.z <= 0.f || wo.z <= 0.f)
            return float3(0.f);

        let config = get_config();
        let latents = get_latents(config, si);
        let result = config.decoder.forward({ material_id, mip_level, latents, wi, wo });
        var f = float3(result[0], result[1], result[2]);
        f = max(f, float3(0.f));
        if (any(isinf(f)) || any(isnan(f)))
            f = float3(0.f);
        return f;
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample result)
    {
        result = {};

        float3 wi = si.shading_frame_ws.to_local(si.wi_ws);
        if (wi.z <= 0.f)
            return false;

        let config = get_config();
        let latents = get_latents(config, si);
        let params = config.sampler.decode_params({ latents, wi });

        float3 wo = config.sampler.sample(wi, next_float3(sg), params);
        result.wo_ws = si.shading_frame_ws.to_global(wo);
        result.pdf = config.sampler.pdf(wi, wo, params);
        if (wo.z <= 0.f || result.pdf <= 0.f)
            return false;

        result.weight = eval(si, result.wo_ws, sg) / result.pdf;
        if (any(isinf(result.weight)) || any(isnan(result.weight)))
            return false;

        // rtcnam: firefly clamp on the indirect-bounce importance weight.
        // model.decoder.output_clamp (bsdf_decoder.py) bounds `eval()`'s
        // numerator; this bounds the full ratio, since a capped bsdf
        // divided by a near-zero trained-Sampler pdf can still explode and
        // compound multiplicatively over path-trace bounces -- confirmed
        // directly: output_clamp alone left the render's max pixel value
        // at ~9e15 despite every single BSDF evaluation being bounded to
        // 100 (rtcnam/harness/bsdf_blowup_probe.py). get_config().
        // sampler_weight_cap is a huge sentinel (no-op) unless
        // model.sampler.weight_cap is set in the config.
        result.weight = min(result.weight, float3(NeuralMaterialConfig_<HASH>.sampler_weight_cap));

        result.flags = BSDFFlags::non_delta_reflection;
        return true;
    }

    public float eval_pdf(const SurfaceInteraction si, const float3 wo_ws)
    {
        float3 wi = si.shading_frame_ws.to_local(si.wi_ws);
        float3 wo = si.shading_frame_ws.to_local(wo_ws);
        if (wi.z <= 0.f || wo.z <= 0.f)
            return 0.f;

        let config = get_config();
        let latents = get_latents(config, si);
        let params = config.sampler.decode_params({ latents, wi });
        return config.sampler.pdf(wi, wo, params);
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};
        result.guide_normal = si.shading_frame_ws.normal;
        result.diffuse_reflection_albedo = float3(0.5f);
        return result;
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si)
    {
        return BSDFFlags::non_delta_reflection;
    }
};

// falcor2 material wrapper that selects the latent mip and creates the runtime
// instance above. <HASH> makes the generated type unique to one architecture.
public struct NeuralMaterial_<HASH> : IMaterial
{
    public typealias MaterialInstance = NeuralMaterialInstance_<HASH>;
    public MaterialHeader header;
    public int material_id;
    public uint64_t config_handle;

    NeuralMaterialConfig_<HASH> get_config()
    {
        BufferHandle handle = { config_handle };
        return handle.load_aligned<NeuralMaterialConfig_<HASH>, 16>(0);
    }

    public MaterialInstance setup_material_instance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        let config = get_config();
        const int texture_resolution = config.latents.get_texture_resolution(material_id);
        const NeuralMaterialGlobals g = g_neural_material_globals;

        int mip_level;
        if (g.fixed_mip_level >= 0)
        {
            mip_level = g.fixed_mip_level;
        }
        else
        {
            float lod = scalar_mip_level_from_footprint(g.duv_dx, g.duv_dy, uint2(texture_resolution));
            mip_level = sample_integer_mip_level(lod, g.mip_sample);
        }
        mip_level = max(0, min(mip_level, NeuralMaterialConfig_<HASH>.num_mip_levels - 1));

        // Decoded once here (reusing the `config` already loaded above for
        // texture_resolution) and cached on the instance -- see the field
        // comment on NeuralMaterialInstance_<HASH>.cached_latents. Skipped
        // entirely when caching is off: this runs before eval/sample/
        // eval_pdf's own `wi.z <= 0` early-outs, so for a cheap dense fetch
        // it is net-negative work on backfacing hits. Compile-time branch.
        float[<NUM_LATENTS>] latents = {};
        if (NeuralMaterialConfig_<HASH>.cache_latents)
            latents = config.latents.get_bilinear_stochastic({ material_id, mip_level, si.uv }, g.texel_sample);
        return NeuralMaterialInstance_<HASH>(material_id, mip_level, g.texel_sample, config_handle, latents);
    }
};
