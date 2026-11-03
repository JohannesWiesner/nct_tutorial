/* One owner for initial, structural, and explicitly requested graph layouts. */
(function () {
    const controllers = new WeakMap();

    function status(message) {
        window.dash_clientside.set_props("layout-status", {children: message});
    }

    function syncPositions(cy) {
        // Layouts do not emit dragfree, which Dash normally uses to save positions.
        // Leave selection under Cytoscape's control rather than replaying flags.
        const elements = cy.elements().map(element => {
            const result = {data: {...element.data()}};
            if (element.isNode()) result.position = {...element.position()};
            return result;
        });
        window.dash_clientside.set_props("cytoscape", {elements});
    }

    function clearOverlaps(nodes) {
        // fCoSE can still leave badge/selection bounds overlapping in dense
        // graphs. Resolve these bounds, not just the 48 px node bodies.
        const boxes = nodes.map(node => {
            const box = node.boundingBox();
            return {node, x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2,
                w: box.w + 8, h: box.h + 8,
                dx: node.position("x") - (box.x1 + box.x2) / 2,
                dy: node.position("y") - (box.y1 + box.y2) / 2};
        });
        for (let pass = 0; pass < 30; pass++) {
            let changed = false;
            for (let i = 0; i < boxes.length; i++) {
                for (let j = i + 1; j < boxes.length; j++) {
                    const a = boxes[i], b = boxes[j];
                    const dx = b.x - a.x, dy = b.y - a.y;
                    const ox = (a.w + b.w) / 2 - Math.abs(dx);
                    const oy = (a.h + b.h) / 2 - Math.abs(dy);
                    if (ox <= 0 || oy <= 0) continue;
                    const axis = ox < oy ? "x" : "y";
                    const distance = axis === "x" ? dx : dy;
                    const shift = ((axis === "x" ? ox : oy) + 0.1) / 2
                        * (distance < 0 ? -1 : 1);
                    a[axis] -= shift;
                    b[axis] += shift;
                    changed = true;
                }
            }
            if (!changed) break;
        }
        // A bounded expansion handles any residual overlaps without an
        // unbounded relaxation loop. Usually the local passes suffice.
        let scale = 1;
        for (let i = 0; i < boxes.length; i++) {
            for (let j = i + 1; j < boxes.length; j++) {
                const a = boxes[i], b = boxes[j];
                const needed = Math.min(
                    (a.w + b.w) / (2 * Math.abs(b.x - a.x)),
                    (a.h + b.h) / (2 * Math.abs(b.y - a.y))
                );
                if (Number.isFinite(needed)) scale = Math.max(scale, needed);
            }
        }
        if (scale > 1) scale *= 1.001;
        for (const box of boxes) {
            const position = {x: box.x * scale + box.dx, y: box.y * scale + box.dy};
            if (!Number.isFinite(position.x) || !Number.isFinite(position.y)) {
                throw new Error("Layout produced non-finite positions");
            }
            box.node.position(position);
        }
    }

    function cancel(controller, publish = true) {
        const pending = controller.pending !== null;
        clearTimeout(controller.pending);
        controller.pending = null;
        const job = controller.active;
        // stop() may emit layoutstop synchronously: revoke ownership first.
        controller.active = null;
        if (job) {
            clearTimeout(job.watchdog);
            if (job.layout) job.layout.stop();
            if (publish && !controller.cy.destroyed()) syncPositions(controller.cy);
        }
        if (publish && (job || pending)) status("Arrangement stopped.");
    }

    function start(controller, animate) {
        controller.pending = null;
        const cy = controller.cy;
        if (cy.destroyed() || !cy.nodes().length) {
            status("");
            return;
        }
        const positions = cy.nodes().map(node => ({node, position: {...node.position()}}));
        const job = {layout: null, watchdog: null};
        controller.active = job;
        function fail() {
            if (controller.active !== job) return;
            cancel(controller, false);
            positions.forEach(({node, position}) => node.position(position));
            syncPositions(cy);
            status("Could not arrange the graph. Please try again.");
        }
        try {
            job.layout = cy.layout({
                name: "fcose", quality: "proof", randomize: true,
                animate, animationDuration: 350, fit: false,
                nodeDimensionsIncludeLabels: true,
                nodeRepulsion: 12000, idealEdgeLength: 120, edgeElasticity: 0.15,
                numIter: 1500, tile: true, packComponents: false,
                tilingPaddingHorizontal: 30, tilingPaddingVertical: 30,
            });
            job.layout.one("layoutstop", function () {
                if (controller.active !== job || cy.destroyed()) return;
                clearTimeout(job.watchdog);
                try {
                    clearOverlaps(cy.nodes());
                } catch (_error) {
                    fail();
                    return;
                }
                controller.active = null;
                window.dash_clientside.nctViewport.fit(cy);
                syncPositions(cy);
                status("Graph arranged.");
            });
            // The solver has an iteration limit; this also bounds animation
            // lifetime. It cannot preempt synchronous solver computation.
            job.watchdog = setTimeout(() => {
                if (controller.active === job) cancel(controller);
            }, 8000);
            job.layout.run();
        } catch (_error) {
            fail();
        }
    }

    function request(controller, animate) {
        cancel(controller, false);
        status("Arranging graph…");
        // Coalesce all add/remove events from a React/Dash element batch.
        controller.pending = setTimeout(() => start(controller, animate), 0);
    }

    function attach(cy) {
        if (controllers.has(cy)) return controllers.get(cy);
        const controller = {cy, pending: null, active: null};
        controllers.set(cy, controller);
        const originalFit = cy.fit;
        // Dash Cytoscape also fits out-of-view updates in componentDidUpdate,
        // independently of autoRefreshLayout. Defer those fits while arranging.
        cy.fit = function (...args) {
            if (controller.pending !== null || controller.active) return cy;
            return originalFit.apply(cy, args);
        };
        cy.on("add remove", () => request(controller, true));
        // Generic pan/zoom/resize events also come from responsive rendering.
        // Cancel on user gestures only; resize finishes against the new canvas.
        cy.on("tapstart grab dragpan scrollzoom pinchzoom", () => cancel(controller));
        cy.on("destroy", () => {
            cancel(controller, false);
            cy.fit = originalFit;
            controllers.delete(cy);
        });
        request(controller, false);
        return controller;
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        nctArrangement: {
            attach,
            arrange: cy => request(attach(cy), true),
            cancel: cy => { if (controllers.has(cy)) cancel(controllers.get(cy)); },
            busy: cy => {
                const controller = controllers.get(cy);
                return !!controller && (controller.pending !== null || !!controller.active);
            },
        },
    });
})();
