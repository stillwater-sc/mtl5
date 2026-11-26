// The animation shell's player (#933): play/pause, step, scrub, a frame
// counter, and prefers-reduced-motion. A figure supplies the number of frames
// and render(frame, t), which draws frame `frame` with t in [0, 1] the
// progress of the tween from the previous frame (t = 1: settled).
//
// The timeline logic is pure (advance, clampFrame) so the tests can run it
// under node; createPlayer binds it to the shell's markup.

/** Clamp a frame index into [0, count - 1]. */
export function clampFrame(frame, count) {
  if (count <= 0) return 0;
  return Math.min(count - 1, Math.max(0, Math.round(frame)));
}

/**
 * Advance a playing timeline by dt milliseconds.
 * state: { frame, elapsed } where elapsed is the time spent on `frame`.
 * Each frame is held for frameMs; its first tweenMs are the tween from the
 * previous frame. Returns { frame, elapsed, t, done }.
 */
export function advance(state, dt, { count, frameMs, tweenMs }) {
  let { frame, elapsed } = state;
  elapsed += dt;
  while (elapsed >= frameMs && frame < count - 1) {
    elapsed -= frameMs;
    frame += 1;
  }
  const done = frame >= count - 1 && elapsed >= tweenMs;
  const t = tweenMs > 0 ? Math.min(1, elapsed / tweenMs) : 1;
  return { frame, elapsed, t, done };
}

/**
 * Where Play starts from. At the last frame it replays from frame 0, shown
 * settled (elapsed = tweenMs). Anywhere else it resumes with the stored
 * elapsed time, so a tween paused halfway finishes instead of jumping.
 */
export function resumeFrom({ frame, elapsed }, { count, tweenMs }) {
  if (frame >= count - 1) return { frame: 0, elapsed: tweenMs, restarted: true };
  return { frame, elapsed, restarted: false };
}

/** Whether the reader asked the system for less motion. */
export function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
}

/**
 * Bind a player to a shell element (AnimationShell.astro's markup).
 *   root:     the shell element
 *   count:    number of frames
 *   render:   (frame, t) => void
 *   label:    (frame) => string, the status line for a frame
 *   frameMs:  how long each frame is held while playing
 *   tweenMs:  how much of that is the tween from the previous frame
 */
export function createPlayer(root, { count, render, label, frameMs = 300, tweenMs = 0 }) {
  const play = root.querySelector('[data-ex-play]');
  const back = root.querySelector('[data-ex-back]');
  const fwd = root.querySelector('[data-ex-fwd]');
  const scrub = root.querySelector('[data-ex-scrub]');
  const counter = root.querySelector('[data-ex-counter]');
  const status = root.querySelector('[data-ex-status]');
  const reduced = prefersReducedMotion();
  const tween = reduced ? 0 : tweenMs;

  let frame = reduced ? count - 1 : 0;
  let elapsed = 0;
  let playing = false;
  let raf = 0;
  let last = 0;

  scrub.max = String(Math.max(0, count - 1));
  scrub.disabled = count <= 1;

  function show(t = 1) {
    render(frame, t);
    scrub.value = String(frame);
    const text = label ? label(frame) : '';
    scrub.setAttribute('aria-valuetext', text || `frame ${frame + 1} of ${count}`);
    counter.textContent = `${frame + 1} / ${count}`;
    if (status) status.textContent = text;
  }

  function setPlaying(on) {
    playing = on;
    // The status line is a polite live region, so a step or a scrub is
    // announced; it is silenced while playing, which would announce every frame.
    if (status) status.setAttribute('aria-live', on ? 'off' : 'polite');
    play.textContent = on ? 'Pause' : 'Play';
    play.setAttribute('aria-pressed', on ? 'true' : 'false');
    root.toggleAttribute('data-playing', on);
    if (on) {
      last = performance.now();
      raf = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(raf);
    }
  }

  function tick(now) {
    const dt = now - last;
    last = now;
    const s = advance({ frame, elapsed }, dt, { count, frameMs, tweenMs: tween });
    frame = s.frame;
    elapsed = s.elapsed;
    show(s.t);
    if (s.done) {
      setPlaying(false);
      return;
    }
    raf = requestAnimationFrame(tick);
  }

  function jump(to) {
    setPlaying(false);
    frame = clampFrame(to, count);
    elapsed = frameMs;
    show(1);
  }

  play.addEventListener('click', () => {
    if (playing) return setPlaying(false);
    const from = resumeFrom({ frame, elapsed }, { count, tweenMs: tween });
    frame = from.frame;
    elapsed = from.elapsed;
    if (from.restarted) show(1);
    setPlaying(true);
  });
  back.addEventListener('click', () => jump(frame - 1));
  fwd.addEventListener('click', () => jump(frame + 1));
  scrub.addEventListener('input', () => jump(Number(scrub.value)));

  show(1);

  // Start once, the first time the figure is mostly on screen, unless the
  // reader prefers reduced motion: then it rests on the final frame.
  if (!reduced && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          if (!playing && frame === 0) setPlaying(true);
        }
      },
      { threshold: 0.5 },
    );
    io.observe(root);
  }

  return { show, jump, get frame() { return frame; } };
}

/**
 * Fetch a figure's trace and start it. Each figure's element carries
 * data-trace (the URL of its JSON, emitted by the build). On failure the
 * figure keeps its text alternative and says the animation could not load.
 */
export function mountAll(selector, setup) {
  for (const root of document.querySelectorAll(selector)) {
    if (root.dataset.exMounted) continue;
    root.dataset.exMounted = '1';
    fetch(root.dataset.trace)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((trace) => setup(root, trace))
      .catch((err) => {
        const status = root.querySelector('[data-ex-status]');
        if (status) status.setAttribute('aria-live', 'polite');
        if (status) status.textContent = `The animation could not load its data (${err.message}); the description below has its numbers.`;
      });
  }
}
