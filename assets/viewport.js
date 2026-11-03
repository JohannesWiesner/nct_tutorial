/* Keep the graph reachable without disabling pan, zoom, or node dragging. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    nctViewport: {
        keepVisible: function (_extent, _elements, fitClicks, arrangeClicks, upload) {
            const noUpdate = window.dash_clientside.no_update;
            // Dash Cytoscape 1.x exposes its current Cytoscape.js instance here.
            const cy = window.cy;
            if (!cy || cy.destroyed() || !cy.container()
                    || cy.container().id !== "cytoscape") {
                return noUpdate;
            }
            const triggered = window.dash_clientside.callback_context.triggered;
            const requested = (id, clicks) => clicks > 0 && triggered.some(
                item => item.prop_id === id + ".n_clicks"
            );
            const fitRequested = requested("fit-graph", fitClicks);
            const arrangement = window.dash_clientside.nctArrangement;
            if (arrangement) {
                arrangement.attach(cy);
                if (upload && triggered.some(item => item.prop_id === "upload-data.contents")) {
                    arrangement.cancel(cy);
                }
                if (requested("arrange-graph", arrangeClicks)) arrangement.arrange(cy);
                if (fitRequested) arrangement.cancel(cy);
                if (arrangement.busy(cy)) return noUpdate;
            }
            window.dash_clientside.nctViewport.recover(cy, fitRequested);
            return noUpdate;
        },
        fit: function (cy) {
            window.dash_clientside.nctViewport.recover(cy, true);
        },
        recover: function (cy, fitRequested = false) {
            const noUpdate = window.dash_clientside.no_update;
            const nodes = cy.nodes(":visible");
            const width = cy.width();
            const height = cy.height();
            if (!nodes.length || width <= 0 || height <= 0) {
                return noUpdate;
            }

            const padding = Math.min(32, width / 4, height / 4);
            const bounds = nodes.boundingBox();
            const fitZoom = Math.min(
                (width - 2 * padding) / Math.max(bounds.w, 1),
                (height - 2 * padding) / Math.max(bounds.h, 1)
            );
            if (!Number.isFinite(fitZoom) || fitZoom <= 0) {
                return noUpdate;
            }
            // Adapt to uploaded/edited graphs, including graphs larger than the
            // initial network. A fixed minimum can prevent large graphs fitting.
            const minZoom = Math.min(0.5, fitZoom / 2);
            cy.minZoom(minZoom);

            // Test individual nodes: the overall bounding box can contain the
            // viewport even when it shows only empty space between components.
            const visibleNode = nodes.some(function (node) {
                const box = node.renderedBoundingBox();
                return box.x2 >= padding && box.x1 <= width - padding
                    && box.y2 >= padding && box.y1 <= height - padding;
            });
            if (fitRequested || !visibleNode || cy.zoom() < minZoom
                    || cy.zoom() > cy.maxZoom()) {
                // Fit changes only the viewport, preserving positions/selection.
                cy.fit(nodes, padding);
            }
            return noUpdate;
        }
    }
});
