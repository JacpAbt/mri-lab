/* vanilla islands (constitution §7: state-in-URL where it matters, ~1KB total,
   zero ambient animation). Three islands: channel highlight, how-to-read
   toggle, g-prefixed room keys. */
(function () {
  // 1. hovering a legend-gauge row isolates that channel on the portrait
  var P = document.querySelector('.portrait');
  if (P) {
    document.querySelectorAll('.g[data-ch]').forEach(function (r) {
      r.addEventListener('pointerenter', function () { P.setAttribute('data-hl', r.dataset.ch); });
      r.addEventListener('pointerleave', function () { P.removeAttribute('data-hl'); });
    });
  }
  // 2. HOW-TO-READ overlay toggle (initial state is server-rendered from ?anno=)
  document.querySelectorAll('.toggle[data-anno-target]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var fig = document.querySelector(btn.dataset.annoTarget);
      if (!fig) return;
      var on = fig.getAttribute('data-anno') === 'on';
      fig.setAttribute('data-anno', on ? 'off' : 'on');
      btn.setAttribute('aria-pressed', String(!on));
    });
  });
  // 3. keyboard: g then a letter jumps rooms (hint in the nav)
  var label = document.body.dataset.subject;
  var map = { q: '/queue', c: '/registry', h: '/' };
  if (label) {
    var base = '/m/' + encodeURIComponent(label);
    map.o = base; map.r = base + '/readiness'; map.f = base + '/fragility';
    map.e = base + '/economics'; map.l = base + '/lineage'; map.p = base + '/preflight';
  }
  var armed = false;
  document.addEventListener('keydown', function (e) {
    if (e.target.matches('input,select,textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    if (!armed) { armed = (e.key === 'g'); return; }
    armed = false;
    if (map[e.key]) location.href = map[e.key];
  });
})();
