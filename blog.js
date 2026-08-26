/* ═══════════════════════════════════════════════════════════════
   blog.js: Interactive charts, TOC polish, scroll animations
   The Wrong Kind of Momentum
   ═══════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    initTocIndicator();
    initStatCounters();

    if (typeof d3 === "undefined") {
      console.warn("[blog.js] D3 not loaded: D3 charts disabled.");
      return;
    }

    initRevealChart();
  }

  /* ────────────────────────────────────────────────────────
     1 · TOC active indicator
     ──────────────────────────────────────────────────────── */
  function initTocIndicator() {
    const toc = document.querySelector("#quarto-sidebar-toc-left #TOC");
    if (!toc) return;

    const links = Array.from(toc.querySelectorAll("a[data-scroll-target]"));
    if (!links.length) return;

    let indicator = document.getElementById("toc-active-indicator");

    if (!indicator) {
      indicator = document.createElement("div");
      indicator.id = "toc-active-indicator";
      toc.appendChild(indicator);
    }

    function getTarget(link) {
      const target = link.getAttribute("data-scroll-target");
      if (!target) return null;

      if (target.startsWith("#")) {
        return document.getElementById(decodeURIComponent(target.slice(1)));
      }

      return document.querySelector(decodeURIComponent(target));
    }

    function setActiveLink(activeLink) {
      links.forEach((link) => link.classList.remove("active"));

      if (!activeLink) {
        indicator.style.opacity = "0";
        return;
      }

      activeLink.classList.add("active");

      const tocRect = toc.getBoundingClientRect();
      const linkRect = activeLink.getBoundingClientRect();
      const top = linkRect.top - tocRect.top + toc.scrollTop;

      indicator.style.top = `${top}px`;
      indicator.style.height = `${activeLink.offsetHeight}px`;
      indicator.style.opacity = "1";
    }

    function updateToc() {
      let current = null;

      for (const link of links) {
        const section = getTarget(link);
        if (!section) continue;

        const rect = section.getBoundingClientRect();

        if (rect.top <= 140) {
          current = link;
        }
      }

      if (!current && links.length && window.scrollY > 0) {
        current = links[0];
      }

      setActiveLink(current);
    }

    let ticking = false;

    function requestUpdate() {
      if (ticking) return;

      ticking = true;
      window.requestAnimationFrame(() => {
        updateToc();
        ticking = false;
      });
    }

    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", requestUpdate);

    links.forEach((link) => {
      link.addEventListener("click", () => {
        window.setTimeout(requestUpdate, 80);
      });
    });

    updateToc();
  }

  /* ────────────────────────────────────────────────────────
     2 · Animated stat counters
     ──────────────────────────────────────────────────────── */
  function initStatCounters() {
    const counters = document.querySelectorAll("[data-count]");
    if (!counters.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.3 }
    );

    counters.forEach((el) => observer.observe(el));
  }

  function animateCounter(el) {
    const target = parseInt(el.dataset.count, 10);
    if (Number.isNaN(target)) return;

    const duration = 1600;
    const startTime = performance.now();
    const formatter = target >= 1000 ? (n) => n.toLocaleString() : (n) => String(n);

    function tick(now) {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 4);
      const current = Math.round(target * eased);

      el.textContent = formatter(current);

      if (t < 1) {
        requestAnimationFrame(tick);
      }
    }

    requestAnimationFrame(tick);
  }

  /* ────────────────────────────────────────────────────────
     3 · Reveal chart
     ──────────────────────────────────────────────────────── */
  // ATE values loaded from Python-generated blog_data.js
  // Run scripts/build_blog_data.py to regenerate

  function initRevealChart() {
    const container = document.getElementById("reveal-chart");
    if (!container) return;

    const margin = { top: 40, right: 40, bottom: 80, left: 80 };
    const width = container.clientWidth || 800;
    const height = 420;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-family", "IBM Plex Sans, sans-serif")
      .style("font-size", "13px")
      .style("font-weight", "700")
      .style("fill", "#17150F")
      .text("Figure 6: Average Treatment Effect of Winning a High-Leverage Point");

    // Domain widened to fit the server-game-point split bar (ATE ~-0.14),
    // which the original [-0.09, 0.20] range (sized for BP/tiebreak only)
    // would clip.
    const y = d3
      .scaleLinear()
      .domain([-0.17, 0.20])
      .range([innerHeight, 0])
      .nice();

    const yAxis = g
      .append("g")
      .attr("class", "axis y-axis")
      .call(
        d3.axisLeft(y).ticks(6).tickFormat((d) => `${d > 0 ? "+" : ""}${d.toFixed(3)}`)
      );

    yAxis.selectAll("line").attr("stroke", "#D9D4C5");
    yAxis.selectAll("path").attr("stroke", "#D9D4C5");
    yAxis
      .selectAll("text")
      .style("font-family", "IBM Plex Mono, monospace")
      .style("font-size", "10.5px")
      .attr("fill", "#6A6658");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -56)
      .attr("x", -innerHeight / 2)
      .attr("text-anchor", "middle")
      .style("font-family", "IBM Plex Sans, sans-serif")
      .style("font-size", "10.5px")
      .style("font-weight", "600")
      .style("letter-spacing", "0.12em")
      .attr("fill", "#6A6658")
      .text("Average Treatment Effect");

    g.append("line")
      .attr("class", "zero-line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", y(0))
      .attr("y2", y(0))
      .attr("stroke", "#17150F")
      .attr("stroke-width", 1.25);

    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(y.ticks(6).filter((d) => d !== 0))
      .join("line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", y)
      .attr("y2", y)
      .attr("stroke", "#D9D4C5")
      .attr("stroke-dasharray", "2,3");

    const barGroup = g.append("g").attr("class", "bars");
    const labelGroup = g.append("g").attr("class", "labels");

    const fillColor = (d) => (d.value >= 0 ? "#2F7A4E" : "#9E3344");

    let currentView = "combined";

    // Split view now carries three pressure types per tour (Break Point,
    // Tiebreak, Server Game Point) rather than the original two; this order
    // fixes each type's slot regardless of the order rows arrive in from
    // blog_data.js.
    const SPLIT_TYPE_ORDER = ["Break Point", "Tiebreak", "Server Game Point"];

    function render(view) {
      currentView = view;
      const data = ATE[view];
      const groupGap = innerWidth / 2;
      const nTypes = SPLIT_TYPE_ORDER.length;
      // Split-view bar width is derived from the container's ACTUAL measured
      // innerWidth, not a fixed guess -- a hardcoded value here (the original
      // bug: 70px, sized for the combined-view's wider container assumption)
      // silently overlapped bars and collided sub-labels once real rendering
      // showed the reveal-section's container is much narrower (~460px inner,
      // not ~720-800px) than a generic dry-run assumes. 0.82 of each tour's
      // half-width slot is allotted to its bars, leaving the remainder as
      // breathing room between tour groups.
      const barWidth = view === "combined"
        ? 120
        : Math.max(28, Math.min(70, (groupGap * 0.82 - (nTypes - 1) * 12) / nTypes));

      let positions;

      if (view === "combined") {
        const xGap = innerWidth / (data.length + 1);
        positions = data.map((d, i) => ({
          ...d,
          key: d.tour,
          x: xGap * (i + 1) - barWidth / 2,
          subLabel: ""
        }));
      } else {
        positions = data.map((d) => {
          const tourIndex = d.tour === "ATP" ? 0 : 1;
          const typeIndex = SPLIT_TYPE_ORDER.indexOf(d.type);
          const groupCenter = groupGap * (tourIndex + 0.5);
          const offset = (typeIndex - (nTypes - 1) / 2) * (barWidth + 12);

          return {
            ...d,
            key: `${d.tour}-${d.type}`,
            x: groupCenter + offset - barWidth / 2,
            subLabel: d.type
          };
        });
      }

      const bars = barGroup.selectAll(".bar").data(positions, (d) => d.key);

      bars.exit()
        .transition()
        .duration(600)
        .ease(d3.easeCubicInOut)
        .attr("y", y(0))
        .attr("height", 0)
        .style("opacity", 0)
        .remove();

      const barsEnter = bars.enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", (d) => d.x)
        .attr("width", barWidth)
        .attr("y", y(0))
        .attr("height", 0)
        .attr("fill", fillColor)
        .style("opacity", 0);

      barsEnter.merge(bars)
        .transition()
        .duration(800)
        .ease(d3.easeCubicInOut)
        .attr("x", (d) => d.x)
        .attr("width", barWidth)
        .attr("y", (d) => (d.value >= 0 ? y(d.value) : y(0)))
        .attr("height", (d) => Math.abs(y(d.value) - y(0)))
        .attr("fill", fillColor)
        .style("opacity", 1);

      const valueLabels = labelGroup
        .selectAll(".val-label")
        .data(positions, (d) => d.key);

      valueLabels.exit().transition().duration(400).style("opacity", 0).remove();

      const valueEnter = valueLabels.enter()
        .append("text")
        .attr("class", "val-label")
        .attr("text-anchor", "middle")
        .style("font-family", "IBM Plex Mono, monospace")
        .style("font-size", "13px")
        .style("font-weight", "500")
        .style("fill", "#17150F")
        .style("opacity", 0);

      valueEnter.merge(valueLabels)
        .transition()
        .duration(800)
        .ease(d3.easeCubicInOut)
        .attr("x", (d) => d.x + barWidth / 2)
        .attr("y", (d) => (d.value >= 0 ? y(d.value) - 10 : y(d.value) + 18))
        .style("opacity", 1)
        .text((d) => `${d.value >= 0 ? "+" : ""}${d.value.toFixed(4)}`);

      const tourData = view === "combined"
        ? positions
        : ["ATP", "WTA"].map((tour) => {
            const xs = positions.filter((d) => d.tour === tour).map((d) => d.x);
            return { tour, x: (Math.min(...xs) + Math.max(...xs)) / 2 + barWidth / 2 };
          });

      const tourLabels = labelGroup
        .selectAll(".tour-label")
        .data(tourData, (d) => d.tour);

      tourLabels.exit().transition().duration(400).style("opacity", 0).remove();

      const tourEnter = tourLabels.enter()
        .append("text")
        .attr("class", "tour-label")
        .attr("text-anchor", "middle")
        .attr("y", innerHeight + 26)
        .style("font-family", "IBM Plex Sans, sans-serif")
        .style("font-size", "12px")
        .style("font-weight", "600")
        .style("letter-spacing", "0.08em")
        .style("fill", "#17150F")
        .style("opacity", 0);

      tourEnter.merge(tourLabels)
        .transition()
        .duration(800)
        .attr("x", (d) => (view === "combined" ? d.x + barWidth / 2 : d.x))
        .style("opacity", 1)
        .text((d) => d.tour);

      const subLabels = labelGroup
        .selectAll(".sub-label")
        .data(view === "split" ? positions : [], (d) => d.key);

      subLabels.exit().transition().duration(300).style("opacity", 0).remove();

      // Sub-label elements are always freshly created here (never re-bound
      // with new text -- exiting the split view removes them entirely, per
      // the data-join above), so per-datum tspan wrapping can live entirely
      // in .enter() without a separate update path. "Server Game Point" is
      // wider than a three-per-tour slot can fit on one line at a readable
      // size (measured: ~100px at 10.5px font vs. a ~55-70px bar slot), so
      // labels with more than two words wrap onto a second line rather than
      // shrinking to the point of illegibility or overflowing into the
      // neighbouring group's label.
      const subLabelsEnter = subLabels.enter()
        .append("text")
        .attr("class", "sub-label")
        .attr("text-anchor", "middle")
        .style("font-family", "IBM Plex Sans, sans-serif")
        .style("font-size", "10.5px")
        .style("font-weight", "500")
        .style("letter-spacing", "0.06em")
        .style("fill", "#6A6658")
        .style("opacity", 0)
        .attr("x", (d) => d.x + barWidth / 2);

      subLabelsEnter.each(function (d) {
        const words = d.subLabel.split(" ");
        const lines = words.length > 2
          ? [words.slice(0, -1).join(" "), words[words.length - 1]]
          : [d.subLabel];
        const startY = innerHeight + (lines.length > 1 ? 40 : 48);
        d3.select(this)
          .selectAll("tspan")
          .data(lines)
          .join("tspan")
          .attr("x", d.x + barWidth / 2)
          .attr("y", (_, i) => startY + i * 13)
          .text((line) => line);
      });

      subLabelsEnter
        .transition()
        .delay(400)
        .duration(500)
        .style("opacity", 1);
    }

    render("combined");

    document.querySelectorAll(".reveal-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".reveal-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        render(btn.dataset.view);
      });
    });

    const section = container.closest(".reveal-section");
    if (section) {
      let autoSwitched = false;

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting || autoSwitched) return;

            autoSwitched = true;

            window.setTimeout(() => {
              if (currentView !== "combined") return;

              document.querySelectorAll(".reveal-btn").forEach((b) => b.classList.remove("active"));

              const splitBtn = document.querySelector('.reveal-btn[data-view="split"]');
              if (splitBtn) splitBtn.classList.add("active");

              render("split");
            }, 900);
          });
        },
        { threshold: 0.5 }
      );

      observer.observe(section);
    }
  }

})();