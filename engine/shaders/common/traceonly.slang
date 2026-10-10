// Trace-only kernel for the "megakernel vs small kernel" experiment: traces
// the rays the path tracer's CAPTURE_RAYS variant recorded, with the same
// traversal code, and does nothing else. Comparing its rays per second (and
// registers, occupancy) with the path tracer's shows how much the big shader
// costs the traversal (Laine, Karras & Aila, "Megakernels Considered
// Harmful", HPG 2013).

#define TRACE_ONLY
#include "raytrace_traversal.slang"

// Results are written so the traversal can't be optimized away: (t or -1, triangle)
[vk::binding(10, 0)] RWStructuredBuffer<float2> traceHits;

// Matches TraceOnlyPushConstants in VulkanRayTracing.cpp
struct TraceOnlyConstants {
    uint queue;    // Which capture queue to trace
    uint anyHit;   // 1 for shadow-ray queues
    uint capacity; // Rays per queue (pixels per frame)
    uint padding;
};
[vk::push_constant] ConstantBuffer<TraceOnlyConstants> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void compMain(uint3 threadId : SV_DispatchThreadID)
{
    uint count = min(captureCounts[pc.queue], pc.capacity);
    if (threadId.x >= count)
        return;

    CapturedRay captured = capturedRays[pc.queue * pc.capacity + threadId.x];
    Ray ray;
    ray.origin = captured.origin.xyz;
    ray.direction = captured.direction.xyz;

    Hit hit;
    bool found = IntersectScene(ray, 0.0, Infinity, pc.anyHit != 0, hit);
    traceHits[threadId.x] = float2(found ? hit.t : -1.0, asfloat(hit.triangle));
}
