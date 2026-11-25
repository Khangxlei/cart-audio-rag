/* demo.js — drives the interactive CART walkthrough.
 *
 * All scoring goes through cart.js, which is a port of the Python. This file
 * only decides what to draw.
 */
(function () {
  'use strict';

  var DATA = null;
  var state = { qid: null, word: null, cond: null };
  var TUNED = { tau: 0.8, lam: 2.0, theta: 0.55, alpha: 0.5, gate: true };

  var el = function (id) { return document.getElementById(id); };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* The payload ships packed — words as one string, probabilities as integer
     percents, alternatives keyed by word index — because the uncompressed form
     is several times larger over the wire. Expand once, here, so the rest of
     the page (and cart.js) sees the ordinary shape. */
  function expandBlock(b) {
    var toks = b.t ? b.t.split(' ') : [];
    var alts = b.a || {};
    return {
      words: toks.map(function (w, i) {
        var rec = [w, (b.p[i] || 0) / 100];
        if (alts[i]) rec.push(alts[i]);
        return rec;
      }),
      conf: b.c
    };
  }

  function unpack(d) {
    d.conditions = d.conditions || ['clean'];
    d.docs = d.docs.map(function (x) {
      var conds = {};
      if (x.cond) {
        Object.keys(x.cond).forEach(function (k) { conds[k] = expandBlock(x.cond[k]); });
      } else {
        conds[d.conditions[0]] = expandBlock(x);   // single-condition payload
      }
      return { id: x.id, title: x.ti || '', gold: x.g || '', conds: conds };
    });
  }

  // ------------------------------------------------------------------ load
  fetch('demo_data.json')
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (d) {
      DATA = d;
      if (d.packed) unpack(d);
      DATA.stopset = Object.create(null);
      (d.stopwords || []).forEach(function (w) { DATA.stopset[w] = 1; });
      DATA.byId = Object.create(null);
      d.docs.forEach(function (x) { DATA.byId[x.id] = x; });
      el('loading').hidden = true;
      el('app').hidden = false;
      el('provenance').textContent = d.provenance || '';
      init();
    })
    .catch(function (e) {
      el('loading').hidden = true;
      el('loadfail').hidden = false;
      if (window.console) console.error('demo data failed to load:', e);
    });

  // ------------------------------------------------------------------ init
  var pending = null;
  function rerankSoon() {
    if (pending) clearTimeout(pending);
    pending = setTimeout(function () { pending = null; rerank(); }, 90);
  }

  function init() {
    ['tau', 'lam', 'theta', 'alpha'].forEach(function (k) {
      el(k).addEventListener('input', function () { syncOuts(); rerankSoon(); });
    });
    el('gate').addEventListener('change', rerank);
    el('reset').addEventListener('click', function () {
      el('tau').value = TUNED.tau; el('lam').value = TUNED.lam;
      el('theta').value = TUNED.theta; el('alpha').value = TUNED.alpha;
      el('gate').checked = TUNED.gate;
      syncOuts(); rerank();
    });
    var cfg = DATA.config || {};
    TUNED = {
      tau: cfg.tau != null ? cfg.tau : 0.8,
      lam: cfg.lam != null ? cfg.lam : 2.0,
      theta: cfg.theta != null ? cfg.theta : 0.55,
      alpha: cfg.alpha != null ? cfg.alpha : 0.5,
      gate: true
    };
    el('tau').value = TUNED.tau; el('lam').value = TUNED.lam;
    el('theta').value = TUNED.theta; el('alpha').value = TUNED.alpha;
    syncOuts();

    state.cond = DATA.conditions[0];
    buildConditionPicker();
    markQueries();
    buildChips();

    el('wordA').addEventListener('input', playground);
    el('wordB').addEventListener('input', playground);
    Array.prototype.forEach.call(document.querySelectorAll('.pill'), function (b) {
      b.addEventListener('click', function () {
        el('wordA').value = b.dataset.a; el('wordB').value = b.dataset.b; playground();
      });
    });

    // open on a question where the ranking actually moves, so the first view
    // shows the mechanism doing something; the chips make the rest visible
    var opener = DATA.queries.filter(function (q) { return q._delta > 0; })[0] ||
                 DATA.queries[0];
    selectQuery(opener.qid);
    playground();
  }

  function syncOuts() {
    el('tauOut').textContent = (+el('tau').value).toFixed(2);
    el('lamOut').textContent = (+el('lam').value).toFixed(2);
    el('thetaOut').textContent = (+el('theta').value).toFixed(2);
    el('alphaOut').textContent = (+el('alpha').value).toFixed(2);
  }

  function knobs() {
    return {
      tau: +el('tau').value,
      lam: +el('lam').value,
      theta: +el('theta').value,
      alpha: +el('alpha').value,
      max_alts: (DATA.config && DATA.config.max_alts) || 3,
      confidence_gate: el('gate').checked,
      use_phonetic: true
    };
  }

  // -------------------------------------------------------------- step 1
  /* Mark every question with what CART actually does to it at the tuned
     settings, so nothing is hidden: ↑ moved the answer up, = unchanged,
     ↓ moved it down. Most are unchanged on clean audio, and that is the
     honest picture. */
  function markQueries() {
    var cfg = {
      tau: TUNED.tau, lam: TUNED.lam, theta: TUNED.theta, alpha: TUNED.alpha,
      max_alts: (DATA.config && DATA.config.max_alts) || 3,
      confidence_gate: true, use_phonetic: true
    };
    DATA.queries.forEach(function (q) {
      var r = evaluate(q, cfg);
      q._base = r[0].rank;
      q._cart = r[2].rank;
      q._delta = r[0].rank - r[2].rank;
    });
  }

  function buildChips() {
    var box = el('queryChips');
    box.innerHTML = '';
    DATA.queries.forEach(function (q) {
      var b = document.createElement('button');
      b.className = 'chip';
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.dataset.qid = q.qid;
      var mark = q._delta > 0 ? '↑ ' : q._delta < 0 ? '↓ ' : '= ';
      var short = q.q.length > 48 ? q.q.slice(0, 46) + '…' : q.q;
      b.textContent = mark + short;
      b.title = q.q + '  —  answer at rank ' + q._base + ' for the baseline, ' +
                q._cart + ' for CART';
      b.addEventListener('click', function () { selectQuery(q.qid); });
      box.appendChild(b);
    });
    var n = DATA.queries.filter(function (q) { return q._delta > 0; }).length;
    var w = DATA.queries.filter(function (q) { return q._delta < 0; }).length;
    var legend = document.createElement('p');
    legend.className = 'chiplegend';
    legend.innerHTML = '↑ CART moved the answer up &middot; = unchanged &middot; ↓ moved it down. ' +
      'On this clean-audio sample ' + n + ' of ' + DATA.queries.length + ' improve, ' + w +
      ' get worse, the rest are unchanged — which is what a +0.007 MRR gain at 4.2% WER looks like ' +
      'query by query. The method is built for the noisy end of the range.';
    box.appendChild(legend);
  }

  /* The payload may carry one acoustic condition or several. With several,
     offer a switcher: watching the gate open as the audio degrades is the whole
     point of the method, and it is the clearest thing the demo can show. */
  function buildConditionPicker() {
    if (!DATA.conditions || DATA.conditions.length < 2) return;
    var host = el('queryChips').parentNode;
    var box = document.createElement('div');
    box.className = 'condpick';
    var label = document.createElement('span');
    label.textContent = 'audio condition:';
    box.appendChild(label);
    DATA.conditions.forEach(function (c) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip cond';
      b.textContent = c + (DATA.wer && DATA.wer[c] != null
        ? '  (' + (DATA.wer[c] * 100).toFixed(1) + '% WER)' : '');
      b.setAttribute('aria-pressed', c === state.cond ? 'true' : 'false');
      b.addEventListener('click', function () {
        state.cond = c;
        retCache = Object.create(null);     // indexes are per-condition
        Array.prototype.forEach.call(box.querySelectorAll('.cond'), function (o) {
          o.setAttribute('aria-pressed', o === b ? 'true' : 'false');
        });
        markQueries();
        buildChips();
        selectQuery(state.qid);
      });
      box.appendChild(b);
    });
    host.insertBefore(box, el('queryChips'));
  }

  function currentQuery() {
    for (var i = 0; i < DATA.queries.length; i++) {
      if (DATA.queries[i].qid === state.qid) return DATA.queries[i];
    }
    return DATA.queries[0];
  }

  function selectQuery(qid) {
    state.qid = qid;
    state.word = null;
    Array.prototype.forEach.call(document.querySelectorAll('.chip'), function (c) {
      c.setAttribute('aria-pressed', c.dataset.qid === qid ? 'true' : 'false');
    });
    var q = currentQuery();
    var doc = DATA.byId[q.gold];
    var cond = state.cond;
    var werPct = ((DATA.wer && DATA.wer[cond] != null) ? DATA.wer[cond] * 100 : 0).toFixed(1);

    var conf = doc.conds[cond].conf;
    el('queryCard').innerHTML =
      '<div class="q">“' + esc(q.q) + '”</div>' +
      '<div class="meta">answer: <b>' + esc((q.answers || []).join(' / ')) + '</b>' +
      ' &middot; answer passage: <b>' + esc(doc && doc.title ? doc.title : q.gold) + '</b>' +
      ' &middot; condition: <b>' + esc(cond) + '</b> (' + werPct + '% WER)' +
      ' &middot; passage confidence c = <b>' + conf.toFixed(3) + '</b>, so the gate (1−c) = <b>' +
      (1 - conf).toFixed(3) + '</b></div>' +
      (q.missing && q.missing.length
        ? '<div class="tagline">lost by the recognizer in this passage: ' +
          q.missing.map(function (m) { return '<span class="missing">' + esc(m) + '</span>'; }).join('') +
          '</div>'
        : '');

    drawTranscript();
    rerank();
  }

  // -------------------------------------------------------------- step 2
  function drawTranscript() {
    var q = currentQuery();
    var doc = DATA.byId[q.gold];
    var cond = state.cond;
    var c = doc.conds[cond];
    var tau = +el('tau').value;
    var missing = Object.create(null);
    (q.missing || []).forEach(function (m) { missing[m] = 1; });
    var qterms = Object.create(null);
    CART.tokenize(q.q).forEach(function (t) {
      if (DATA.stopset[t] !== 1 && CART.indexable(t)) qterms[t] = 1;
    });

    var html = '';
    c.words.forEach(function (w, i) {
      var cls = 'tw', p = w[1], hasAlt = (w[2] || []).length > 0;
      if (p < tau && hasAlt) cls += ' alt';
      else if (p < tau) cls += ' lo';
      else if (p < 0.9) cls += ' md';
      if (qterms[w[0]]) cls += ' qt';
      html += '<span class="' + cls + '" data-i="' + i + '" title="p = ' + p.toFixed(2) + '">' +
              esc(w[0]) + '</span> ';
    });
    // the terms the transcript lost are shown separately: they are, by
    // definition, not in the transcript to highlight
    el('transcript').innerHTML = html;

    Array.prototype.forEach.call(el('transcript').querySelectorAll('.tw'), function (s) {
      s.addEventListener('click', function () { showWord(+s.dataset.i); });
    });

    el('goldText').textContent = doc.gold;
    el('wordInfo').innerHTML =
      'Click a word above to see its confidence and the spellings competing beams proposed.' +
      (q.missing && q.missing.length
        ? ' <em>Note the query terms ' +
          q.missing.map(function (m) { return '<code>' + esc(m) + '</code>'; }).join(', ') +
          ' are nowhere in this transcript — that is the failure CART has to recover from.</em>'
        : '');
  }

  function showWord(i) {
    var q = currentQuery();
    var c = DATA.byId[q.gold].conds[state.cond];
    var w = c.words[i];
    state.word = i;
    Array.prototype.forEach.call(el('transcript').querySelectorAll('.tw'), function (s) {
      s.classList.toggle('sel', +s.dataset.i === i);
    });
    var tau = +el('tau').value;
    var ph = DATA.phones[w[0]];
    var out = '<span class="wtag">' + esc(w[0]) + '</span> &nbsp; confidence <b>' +
      w[1].toFixed(2) + '</b> &nbsp; ' +
      (w[1] < tau
        ? '<span style="color:var(--accent)">below τ = ' + tau.toFixed(2) + ', so CART expands it</span>'
        : '<span style="color:var(--fg-faint)">at or above τ, so CART leaves it alone</span>');
    if (ph) out += ' &nbsp; <span class="phonepill">' + esc(ph.join(' ')) + '</span>';

    var alts = w[2] || [];
    if (alts.length) {
      out += '<div class="alts">competing beams proposed: ' +
        alts.map(function (a) {
          var ap = DATA.phones[a];
          var sim = (ph && ap) ? CART.phoneticSimilarity(ph, ap) : null;
          return '<span class="altpill">' + esc(a) + '</span>' +
                 (sim != null ? '<span class="phonepill">sounds ' + (sim * 100).toFixed(0) + '% alike</span> ' : ' ');
        }).join('') + '</div>';
    } else {
      out += '<div class="alts"><span class="none" style="color:var(--fg-faint)">no competing spellings at this position' +
             (w[1] < tau ? ' — only the phonetic channel can help here' : '') + '</span></div>';
    }
    el('wordInfo').innerHTML = out;
  }

  // -------------------------------------------------------------- step 3+4
  /* An index does not depend on the query, only on the knobs that change what
     gets indexed (τ, α, θ). Building one over 825 passages costs ~100 ms, so
     they are cached and reused across queries; λ and the gate affect only
     scoring and never trigger a rebuild. */
  var SYSTEMS = [
    { key: '1best', label: 'BM25 / 1-best', note: 'the standard cascade' },
    { key: 'nbest_uniform', label: 'N-best uniform', note: 'all alternatives, untargeted' },
    { key: 'cart', label: 'CART', note: 'targeted + phonetic fallback', ours: true }
  ];
  var retCache = Object.create(null);

  function retrieverFor(sys, cfg) {
    // only the knobs that actually change the index belong in the key
    var key = state.cond + '|' + sys.key + '|' +
      (sys.key === '1best' ? '' :
       sys.key === 'nbest_uniform' ? cfg.alpha :
       cfg.tau + ',' + cfg.alpha + ',' + cfg.theta);
    if (!retCache[key]) {
      retCache[key] = CART.buildRetriever(DATA, state.cond, sys.key, cfg);
    }
    var r = retCache[key];
    r.cfg = cfg;            // λ and the gate are read at scoring time
    return r;
  }

  function evaluate(q, cfg) {
    return SYSTEMS.map(function (s) {
      var ret = retrieverFor(s, cfg);
      var sc = CART.score(ret, q.q, DATA);
      var goldIdx = ret.bm25.docIds.indexOf(q.gold);
      return { sys: s, ret: ret, sc: sc, goldIdx: goldIdx,
               rank: sc.order.indexOf(goldIdx) + 1 };
    });
  }

  function rerank() {
    var q = currentQuery();
    var cfg = knobs();
    var results = evaluate(q, cfg);

    drawChannels(results, q, cfg);
    drawRanks(results, q);
    drawTranscriptClasses();
  }

  function drawTranscriptClasses() {
    // τ changes which words are outlined, so redraw shading without losing the
    // selected word
    var keep = state.word;
    drawTranscript();
    if (keep != null) showWord(keep);
  }

  function drawChannels(results, q, cfg) {
    var cart = results[2];
    var goldIdx = cart.goldIdx;
    var terms = cart.sc.terms;

    // channel 1: plain text matches against the gold passage
    var oneBest = results[0];
    var textRows = terms.map(function (t) {
      var hits = oneBest.sc.textHits[t] || [];
      var inGold = hits.some(function (h) { return h.doc === oneBest.goldIdx; });
      return '<li><span class="term">' + esc(t) + '</span>' +
        '<span class="num-r ' + (inGold ? 'hit' : 'nohit') + '">' +
        (inGold ? 'found' : 'missing') + '</span></li>';
    }).join('');

    // channel 2: did expansion postings put the term in the gold passage?
    var cartDoc = cart.ret.docs[goldIdx];
    var altSet = Object.create(null);
    (cartDoc.altTokens || []).forEach(function (a) { altSet[a] = 1; });
    var oneSet = Object.create(null);
    cartDoc.tokens.forEach(function (t) { oneSet[t] = 1; });
    var recovered = terms.filter(function (t) { return !oneSet[t] && altSet[t]; });
    var altRows = recovered.length
      ? recovered.map(function (t) {
          return '<li><span class="term">' + esc(t) + '</span>' +
                 '<span class="num-r hit">recovered, weight ' + cfg.alpha.toFixed(2) + '</span></li>';
        }).join('')
      : '<li class="none">No query term was recovered from a beam alternative in this passage. ' +
        (Object.keys(altSet).length
          ? 'It does carry ' + Object.keys(altSet).length + ' expansion postings, but none match this query.'
          : 'No word here fell below τ with an alternative attached.') + '</li>';

    // channel 3: phonetic
    var phRows = '<li class="none">The phonetic channel contributed nothing for this query.</li>';
    if (cart.sc.phon && cart.sc.phon.detail.length) {
      var rows = cart.sc.phon.detail.map(function (d) {
        var v = d.best[goldIdx];
        var who = d.who[goldIdx];
        if (!(v > 0) || !who) return '';
        return '<li><span class="term">' + esc(d.term) + '</span> → ' +
          '<span class="term" style="color:var(--accent)">' + esc(who.term) + '</span>' +
          '<span class="num-r hit">' + (who.sim * 100).toFixed(0) + '% alike</span>' +
          '<div class="bar" style="width:' + Math.min(100, v * 100).toFixed(0) + '%"></div></li>';
      }).filter(Boolean).join('');
      var suppressed = cart.sc.phon.detail.reduce(function (a, d) { return a + d.suppressed; }, 0);
      if (rows) {
        phRows = rows + (suppressed
          ? '<li class="none">Suppressed in ' + suppressed +
            ' passage(s) whose transcript already contained the term — the channel only fires where matching failed.</li>'
          : '');
      }
    }

    el('channels').innerHTML =
      '<div class="ch"><h4><span class="dot d1"></span>1 · Text (1-best)</h4>' +
      '<p class="cdesc">Ordinary BM25 over what the recognizer wrote.</p><ul>' +
      (textRows || '<li class="none">no content terms in this query</li>') + '</ul></div>' +

      '<div class="ch"><h4><span class="dot d2"></span>2 · Alternative spellings</h4>' +
      '<p class="cdesc">Beam alternatives for words below τ = ' + cfg.tau.toFixed(2) +
      ', added as extra postings.</p><ul>' + altRows + '</ul></div>' +

      '<div class="ch"><h4><span class="dot d3"></span>3 · Phonetic fallback</h4>' +
      '<p class="cdesc">Query term matched to a low-confidence term that <em>sounds</em> like it (θ = ' +
      cfg.theta.toFixed(2) + ').</p><ul>' + phRows + '</ul></div>';
  }

  function drawRanks(results, q) {
    var html = results.map(function (r) {
      var top = r.sc.order.slice(0, 6);
      var max = r.sc.total[r.sc.order[0]] || 1;
      var items = top.map(function (di, i) {
        var d = DATA.byId[r.ret.bm25.docIds[di]];
        var isGold = di === r.goldIdx;
        var sc = r.sc.total[di];
        return '<li class="' + (isGold ? 'isgold' : '') + '">' +
          '<span class="pos">' + (i + 1) + '.</span>' +
          '<span class="ttl">' + esc(d && d.title ? d.title : r.ret.bm25.docIds[di]) +
          (isGold ? ' ← answer' : '') + '</span>' +
          '<span class="sc">' + sc.toFixed(3) + '</span></li>';
      }).join('');
      var found = r.rank > 0;
      return '<div class="rank' + (r.sys.ours ? ' ours' : '') + '">' +
        '<div class="rhead"><h4>' + esc(r.sys.label) + '</h4>' +
        '<div class="gold">' + esc(r.sys.note) + ' &middot; answer at rank <b>' +
        (found ? r.rank : '—') + '</b>' +
        (found ? ' &middot; RR ' + (1 / r.rank).toFixed(3) : '') + '</div></div>' +
        '<ol>' + items + '</ol></div>';
    }).join('');
    el('ranks').innerHTML = html;

    var base = results[0].rank, ours = results[2].rank;
    var note = el('rankNote');
    if (base && ours && ours < base) {
      note.hidden = false;
      note.className = 'note good';
      note.innerHTML = '<p><span class="lbl">CART moved the answer up, ' +
        'from rank ' + base + ' to rank ' + ours + '.</span> Step 3 shows which channel did it.</p>';
    } else if (base && ours && ours > base) {
      note.hidden = false;
      note.className = 'note warn';
      if (!el('gate').checked) {
        note.innerHTML = '<p><span class="lbl">This is the confidence gate being switched off, ' +
          'and it is the most important result in the paper.</span> The answer has fallen from rank ' +
          base + ' to rank ' + ours + '.</p>' +
          '<p>Without the <code>(1 − c)</code> factor the phonetic channel is trusted equally on every ' +
          'passage, including the ones the recognizer got right, so phonetic near-matches start ' +
          'outscoring genuine text matches. In the full experiment this costs 0.141 MRR at 37% WER — ' +
          'an order of magnitude more than removing either retrieval channel. Tick the gate again to ' +
          'restore it.</p>';
      } else if (+el('lam').value > 6) {
        note.innerHTML = '<p><span class="lbl">λ is high enough to drown the text score.</span> ' +
          'The answer sits at rank ' + ours + ' against ' + base + ' for the baseline. The measured ' +
          'λ sweep does the same thing: at 37% WER, λ = 8 collapses MRR to 0.291 and λ = 16 to 0.062. ' +
          'The phonetic channel has to be weighted, not merely switched on.</p>';
      } else {
        note.innerHTML = '<p><span class="lbl">On this query CART ranks the answer lower</span> — ' +
          'rank ' + ours + ' against ' + base + ' for the baseline. Expansion adds evidence to other ' +
          'passages too, and it does not always pay off. The reported result is a mean over 4,021 ' +
          'queries, not a guarantee on any single one.</p>';
      }
    } else if (base && ours && ours === base) {
      var q2 = currentQuery();
      var c2 = DATA.byId[q2.gold].conds[state.cond].conf;
      note.hidden = false;
      note.className = 'note';
      note.innerHTML = '<p><span class="lbl">Same rank for all three systems here</span> (' + ours +
        '), and on clean audio that is the expected answer rather than a disappointment.</p>' +
        '<p>This passage has confidence <b>c = ' + c2.toFixed(3) + '</b>, so the gate <code>(1 − c)</code> ' +
        'is <b>' + (1 - c2).toFixed(3) + '</b> — the phonetic channel is scaled down to near nothing, ' +
        '<em>by design</em>. That is exactly why the measured gain at 4.2% WER is only +0.007 MRR, ' +
        'and why CART separates from the baseline at 37% WER, where confidence falls and the gate opens. ' +
        'To see the machinery act, untick the confidence gate or push λ up — then watch the ranking move.</p>';
    } else {
      note.hidden = true;
    }
  }

  // -------------------------------------------------------------- step 5
  function playground() {
    var a = CART.normalize(el('wordA').value || '');
    var b = CART.normalize(el('wordB').value || '');
    var pa = DATA.phones[a], pb = DATA.phones[b];
    var out = el('pgOut');

    if (!a || !b) { out.innerHTML = '<span class="verdict">Enter two words.</span>'; return; }
    if (!pa || !pb) {
      var miss = [];
      if (!pa) miss.push(a);
      if (!pb) miss.push(b);
      out.innerHTML = '<span class="verdict">No pronunciation shipped for ' +
        miss.map(function (m) { return '<code>' + esc(m) + '</code>'; }).join(' and ') +
        '. This page carries phonemes only for the vocabulary of its sample — ' +
        'grapheme-to-phoneme needs a model that cannot run in a static page. Try one of the pairs below.</span>';
      return;
    }
    var sim = CART.phoneticSimilarity(pa, pb);
    var dist = CART.weightedEdit(pa, pb);
    var verdict = sim >= 0.8 ? 'highly confusable — the kind of pair CART is built to bridge'
      : sim >= 0.55 ? 'similar enough to pass the default θ = 0.55'
      : 'not confusable; below θ, so CART would not link these';
    out.innerHTML =
      '<div class="big" style="color:' + (sim >= 0.55 ? 'var(--good)' : 'var(--fg-muted)') + '">' +
      sim.toFixed(2) + '</div>' +
      '<div class="simbar"><i style="width:' + (sim * 100).toFixed(0) + '%"></i></div>' +
      '<div><span class="phonepill">' + esc(pa.join(' ')) + '</span> &nbsp;vs&nbsp; ' +
      '<span class="phonepill">' + esc(pb.join(' ')) + '</span></div>' +
      '<div class="verdict" style="margin-top:9px">weighted edit cost ' + dist.toFixed(2) +
      ' &middot; ' + verdict + '</div>';
  }
})();
