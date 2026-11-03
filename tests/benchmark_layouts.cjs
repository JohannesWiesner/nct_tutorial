/* Optional reproducible comparison; not a browser performance benchmark. */
const assert = require("node:assert/strict");
const {performance} = require("node:perf_hooks");
const {graph, fixture, network, overlaps} = require("./graph_test_helpers.cjs");

async function main() {
    const cases = [
        ["tutorial", fixture.elements],
        ["overlapping", network(30)], ["spread", network(30)],
        ["disconnected", network(30, "disconnected")],
        ["isolated", network(12, "isolated")], ["single", network(1)],
        ["dense", network(150)],
    ];
    cases[1][1].nodes.forEach(node => { node.position = {x: 0, y: 0}; });
    for (const [name, elements] of cases) {
        for (const algorithm of ["cose", "fcose", "arrange"]) {
            const cleanups = [];
            const {cy, guard, settled, events} = graph({after: fn => cleanups.push(fn)}, elements);
            const random = Math.random;
            let seed = 46;
            Math.random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296);
            try {
                const start = performance.now();
                if (algorithm === "arrange") {
                    guard(); await settled();
                    assert.equal(events.status, "Graph arranged.");
                } else {
                    cy.layout({
                        name: algorithm, animate: false, fit: false, randomize: true,
                        quality: "proof", nodeDimensionsIncludeLabels: true,
                        nodeRepulsion: 12000, idealEdgeLength: 120,
                        edgeElasticity: algorithm === "cose" ? 32 : 0.15,
                        numIter: 1500, tile: true, packComponents: false,
                        tilingPaddingHorizontal: 30, tilingPaddingVertical: 30,
                    }).run();
                }
                const bounds = cy.nodes().boundingBox();
                console.log(JSON.stringify({
                    case: name, algorithm, nodes: cy.nodes().length, edges: cy.edges().length,
                    ms: Math.round(performance.now() - start), badgeOverlaps: overlaps(cy),
                    width: Math.round(bounds.w), height: Math.round(bounds.h),
                }));
            } finally {
                Math.random = random;
                cleanups.forEach(fn => fn());
            }
        }
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
