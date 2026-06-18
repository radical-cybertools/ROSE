// ROSE docs — subtle scroll-reveal for a more fluid reading experience.
// Pairs with the .rose-reveal / .rose-revealed rules in styles/custom.css.
(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) return;

  const SELECTOR =
    ".md-content__inner > h2, .md-content__inner > h3, .md-content__inner > p, " +
    ".md-content__inner > table, .md-content__inner > .admonition, " +
    ".md-content__inner > pre, .md-content__inner > .highlight";

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("rose-revealed");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.05, rootMargin: "0px 0px -40px 0px" }
  );

  const observe = () => {
    document.querySelectorAll(SELECTOR).forEach((el) => {
      el.classList.add("rose-reveal");
      observer.observe(el);
    });
  };

  // mkdocs-material's instant navigation swaps content without a full page
  // load, so re-run on each page change via its document$ observable.
  if (typeof document$ !== "undefined") {
    document$.subscribe(observe);
  } else {
    document.addEventListener("DOMContentLoaded", observe);
  }
})();
