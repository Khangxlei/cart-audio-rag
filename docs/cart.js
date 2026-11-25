/* cart.js — a faithful browser port of the CART scorer.
 *
 * This mirrors cart/index.py, cart/retrieval.py and cart/phonetics.py so the
 * demo page computes real scores rather than showing canned output. Any change
 * to the Python should be mirrored here. The one thing the browser cannot do is
 * grapheme-to-phoneme conversion, so phoneme strings are precomputed and
 * shipped in demo_data.json.
 *
 * Deliberately dependency-free and synchronous: the sample is small enough
 * (~60 passages) that a full rescore takes a few milliseconds.
 */

(function (global) {
  'use strict';

  var MAX_TERM_LEN = 25;
  var PUNCT = /[^\w']+/g;

  function normalize(tok) {
    return tok.toLowerCase().replace(PUNCT, '').replace(/^'+|'+$/g, '');
  }

  function tokenize(text) {
    var out = [];
    (text || '').split(/\s+/).forEach(function (w) {
      var t = normalize(w);
      if (t) out.push(t);
    });
    return out;
  }

  function indexable(t) { return t.length > 1 && t.length <= MAX_TERM_LEN; }

  // ---------------------------------------------------------------- phonetics
  // (place, manner, voiced) on coarse ordinal scales
  var CONS = {
    P: [1, 1, 0], B: [1, 1, 1], T: [3, 1, 0], D: [3, 1, 1],
    K: [6, 1, 0], G: [6, 1, 1],
    F: [2, 2, 0], V: [2, 2, 1], TH: [3, 2, 0], DH: [3, 2, 1],
    S: [3, 2, 0], Z: [3, 2, 1], SH: [4, 2, 0], ZH: [4, 2, 1],
    HH: [7, 2, 0],
    CH: [4, 3, 0], JH: [4, 3, 1],
    M: [1, 4, 1], N: [3, 4, 1], NG: [6, 4, 1],
    L: [3, 5, 1], R: [4, 5, 1], W: [1, 6, 1], Y: [5, 6, 1]
  };
  // (height, backness, rounded, diphthong)
  var VOW = {
    IY: [3, 0, 0, 0], IH: [2, 0, 0, 0], EY: [2, 0, 0, 1], EH: [1, 0, 0, 0],
    AE: [0, 0, 0, 0], AA: [0, 2, 0, 0], AO: [1, 2, 1, 0], OW: [2, 2, 1, 1],
    UH: [2, 2, 1, 0], UW: [3, 2, 1, 0], AH: [1, 1, 0, 0], ER: [1, 1, 0, 0],
    AY: [0, 1, 0, 1], AW: [0, 1, 1, 1], OY: [1, 2, 1, 1]
  };
  var CHEAP_INDEL = { AH: 1, ER: 1, Y: 1, W: 1, HH: 1 };

  var subCache = Object.create(null);

  function phoneSubCost(a, b) {
    if (a === b) return 0;
    var key = a + '\u0000' + b;
    var hit = subCache[key];
    if (hit !== undefined) return hit;
    var cost;
    if (VOW[a] && VOW[b]) {
      var fa = VOW[a], fb = VOW[b];
      var dv = Math.abs(fa[0] - fb[0]) / 3 + Math.abs(fa[1] - fb[1]) / 2 +
               Math.abs(fa[2] - fb[2]) + Math.abs(fa[3] - fb[3]);
      cost = 0.25 + 0.55 * Math.min(dv / 4.0, 1.0);
    } else if (CONS[a] && CONS[b]) {
      var ca = CONS[a], cb = CONS[b];
      var dc = Math.abs(ca[0] - cb[0]) / 6 + (ca[1] === cb[1] ? 0 : 1) +
               Math.abs(ca[2] - cb[2]);
      cost = 0.25 + 0.65 * Math.min(dc / 3.0, 1.0);
    } else {
      cost = 1.0;   // vowel against consonant: maximally different
    }
    subCache[key] = cost;
    return cost;
  }

  function indelCost(p) { return CHEAP_INDEL[p] ? 0.55 : 0.9; }

  function sumIndel(seq) {
    var s = 0;
    for (var i = 0; i < seq.length; i++) s += indelCost(seq[i]);
    return s;
  }

  function weightedEdit(a, b) {
    if (a.length === b.length) {
      var same = true;
      for (var q = 0; q < a.length; q++) if (a[q] !== b[q]) { same = false; break; }
      if (same) return 0;
    }
    if (!a.length || !b.length) return sumIndel(a.length ? a : b);
    var prev = new Float64Array(b.length + 1);
    for (var j = 1; j <= b.length; j++) prev[j] = prev[j - 1] + indelCost(b[j - 1]);
    var cur = new Float64Array(b.length + 1);
    for (var i = 1; i <= a.length; i++) {
      cur[0] = prev[0] + indelCost(a[i - 1]);
      for (var k = 1; k <= b.length; k++) {
        cur[k] = Math.min(
          prev[k] + indelCost(a[i - 1]),
          cur[k - 1] + indelCost(b[k - 1]),
          prev[k - 1] + phoneSubCost(a[i - 1], b[k - 1])
        );
      }
      var t = prev; prev = cur; cur = t;
    }
    return prev[b.length];
  }

  function phoneticSimilarity(a, b) {
    if (!a.length && !b.length) return 1.0;
    var denom = Math.max(sumIndel(a), sumIndel(b), 1e-9);
    return Math.max(0.0, 1.0 - weightedEdit(a, b) / denom);
  }

  // ------------------------------------------------------------------- BM25
  function counter(list) {
    var m = Object.create(null);
    for (var i = 0; i < list.length; i++) m[list[i]] = (m[list[i]] || 0) + 1;
    return m;
  }

  function BM25Index(docs, k1, b, alpha) {
    this.docs = docs;
    this.k1 = k1; this.b = b; this.alpha = alpha;
    this.docIds = docs.map(function (d) { return d.id; });
    this.N = docs.length;
    this.dl = new Float64Array(this.N);
    var total = 0;
    for (var i = 0; i < this.N; i++) {
      this.dl[i] = Math.max(docs[i].tokens.length, 1);
      total += this.dl[i];
    }
    this.avgdl = total / Math.max(this.N, 1);

    var postings = Object.create(null), df = Object.create(null);
    for (var d = 0; d < this.N; d++) {
      var prim = counter(docs[d].tokens);
      var alts = counter(docs[d].altTokens || []);
      var seen = Object.create(null), t;
      for (t in prim) {
        (postings[t] || (postings[t] = Object.create(null)))[d] = prim[t];
        seen[t] = 1;
      }
      for (t in alts) {
        var p = postings[t] || (postings[t] = Object.create(null));
        p[d] = (p[d] || 0) + alpha * alts[t];
        seen[t] = 1;
      }
      for (t in seen) df[t] = (df[t] || 0) + 1;
    }
    this.postings = Object.create(null);
    for (var term in postings) {
      var ids = [], tfs = [];
      for (var key in postings[term]) { ids.push(+key); tfs.push(postings[term][key]); }
      this.postings[term] = { ids: ids, tfs: tfs };
    }
    this.idf = Object.create(null);
    for (var tm in df) {
      this.idf[tm] = Math.log(1 + (this.N - df[tm] + 0.5) / (df[tm] + 0.5));
    }
    this._denom = new Float64Array(this.N);
    for (var z = 0; z < this.N; z++) {
      this._denom[z] = this.k1 * (1 - this.b + this.b * this.dl[z] / this.avgdl);
    }
  }

  BM25Index.prototype.termIdf = function (t) {
    var v = this.idf[t];
    return v === undefined ? Math.log(1 + (this.N + 0.5) / 0.5) : v;
  };

  /* Returns {scores: Float64Array, hits: {term: [{doc, tf, contrib}]}} so the
     demo can show which term contributed what, not just a total. */
  BM25Index.prototype.scoreExplained = function (queryTerms) {
    var s = new Float64Array(this.N);
    var hits = Object.create(null);
    for (var i = 0; i < queryTerms.length; i++) {
      var t = queryTerms[i], p = this.postings[t];
      if (!p) continue;
      var idf = this.idf[t], rows = [];
      for (var j = 0; j < p.ids.length; j++) {
        var d = p.ids[j], tf = p.tfs[j];
        var c = idf * (tf * (this.k1 + 1)) / (tf + this._denom[d]);
        s[d] += c;
        rows.push({ doc: d, tf: tf, contrib: c });
      }
      hits[t] = rows;
    }
    return { scores: s, hits: hits };
  };

  BM25Index.prototype.contains = function (t) {
    var m = new Uint8Array(this.N), p = this.postings[t];
    if (p) for (var i = 0; i < p.ids.length; i++) m[p.ids[i]] = 1;
    return m;
  };

  // --------------------------------------------------------- phonetic index
  function bigrams(ph) {
    if (ph.length < 2) return ph.length ? [ph[0] + '|#'] : [];
    var out = [];
    for (var i = 0; i + 1 < ph.length; i++) out.push(ph[i] + '|' + ph[i + 1]);
    return out;
  }

  function PhoneticIndex(docs, phones, theta) {
    this.phonesOf = phones;
    this.theta = theta;
    this.N = docs.length;
    this.termDocs = Object.create(null);
    for (var i = 0; i < docs.length; i++) {
      var lc = docs[i].lowconf || [];
      for (var j = 0; j < lc.length; j++) {
        var t = lc[j];
        if (!indexable(t)) continue;
        (this.termDocs[t] || (this.termDocs[t] = [])).push(i);
      }
    }
    this.phones = Object.create(null);
    this.bigrams = Object.create(null);
    for (var term in this.termDocs) {
      var ph = phones[term] || [];
      this.phones[term] = ph;
      var bg = bigrams(ph);
      for (var b = 0; b < bg.length; b++) {
        (this.bigrams[bg[b]] || (this.bigrams[bg[b]] = Object.create(null)))[term] = 1;
      }
    }
    this._cache = Object.create(null);
  }

  PhoneticIndex.prototype.match = function (term) {
    if (this._cache[term]) return this._cache[term];
    var ph = this.phonesOf[term] || [];
    var cands = Object.create(null);
    var bg = bigrams(ph);
    for (var i = 0; i < bg.length; i++) {
      var bucket = this.bigrams[bg[i]];
      if (bucket) for (var t in bucket) cands[t] = 1;
    }
    var out = [];
    for (var cand in cands) {
      if (cand === term) continue;
      var tp = this.phones[cand] || [];
      if (Math.abs(tp.length - ph.length) > 3) continue;
      var s = phoneticSimilarity(ph, tp);
      if (s >= this.theta) out.push({ term: cand, sim: s });
    }
    out.sort(function (a, b) { return b.sim - a.sim; });
    this._cache[term] = out;
    return out;
  };

  /* IDF-weighted, per-query-term max phonetic match, suppressed wherever the
     1-best already contains the term. Returns per-term detail for the UI. */
  PhoneticIndex.prototype.scoreExplained = function (queryTerms, idfFn, containsFn) {
    var s = new Float64Array(this.N);
    var detail = [];
    for (var i = 0; i < queryTerms.length; i++) {
      var q = queryTerms[i];
      var m = this.match(q);
      if (!m.length) continue;
      var w = idfFn(q);
      var best = new Float64Array(this.N);
      var who = new Array(this.N);
      for (var k = 0; k < m.length; k++) {
        var v = (m[k].sim - this.theta) / Math.max(1e-6, 1 - this.theta);
        var dl = this.termDocs[m[k].term];
        for (var j = 0; j < dl.length; j++) {
          if (v > best[dl[j]]) { best[dl[j]] = v; who[dl[j]] = m[k]; }
        }
      }
      var mask = containsFn(q), suppressed = 0;
      for (var z = 0; z < this.N; z++) {
        if (mask[z] && best[z] > 0) { best[z] = 0; who[z] = null; suppressed++; }
        s[z] += w * best[z];
      }
      detail.push({ term: q, idf: w, matches: m, best: best, who: who, suppressed: suppressed });
    }
    return { scores: s, detail: detail };
  };

  // ------------------------------------------------------------- documents
  /* mode: 'oracle' | '1best' | 'nbest_uniform' | 'cart' */
  function buildDocuments(data, cond, mode, tau, maxAlts) {
    return data.docs.map(function (d) {
      var c = d.conds[cond] || d.conds[data.conditions[0]];
      if (mode === 'oracle') {
        return { id: d.id, tokens: tokenize(d.gold), altTokens: [], lowconf: [], conf: 1.0 };
      }
      var toks = c.words.map(function (w) { return w[0]; });
      if (mode === '1best') {
        return { id: d.id, tokens: toks, altTokens: [], lowconf: [], conf: c.conf };
      }
      if (mode === 'nbest_uniform') {
        // every alternative, with no confidence targeting
        var base = Object.create(null);
        toks.forEach(function (t) { base[t] = 1; });
        var all = [];
        c.words.forEach(function (w) {
          (w[2] || []).forEach(function (a) { if (!base[a]) all.push(a); });
        });
        return { id: d.id, tokens: toks, altTokens: all, lowconf: [], conf: c.conf };
      }
      // cart: expand only words the recognizer was unsure about
      var alts = [], low = Object.create(null);
      c.words.forEach(function (w) {
        if (w[1] >= tau) return;
        low[w[0]] = 1;
        (w[2] || []).slice(0, maxAlts).forEach(function (a) { alts.push(a); low[a] = 1; });
      });
      return { id: d.id, tokens: toks, altTokens: alts, lowconf: Object.keys(low), conf: c.conf };
    });
  }

  // -------------------------------------------------------------- retriever
  function buildRetriever(data, cond, mode, cfg) {
    var docs = buildDocuments(data, cond, mode, cfg.tau, cfg.max_alts);
    var useAlpha = (mode === 'cart' || mode === 'nbest_uniform') ? cfg.alpha : 0.0;
    var bm25 = new BM25Index(docs, data.bm25.k1, data.bm25.b, useAlpha);
    var phon = null, ref = null;
    if (mode === 'cart' && cfg.use_phonetic !== false) {
      phon = new PhoneticIndex(docs, data.phones, cfg.theta);
      ref = useAlpha === 0.0
        ? bm25
        : new BM25Index(docs, data.bm25.k1, data.bm25.b, 0.0);
    }
    return { mode: mode, docs: docs, bm25: bm25, phon: phon, ref: ref, cfg: cfg };
  }

  /* Full score with an explanation of every contribution. */
  function score(ret, question, data) {
    var terms = tokenize(question).filter(function (t) {
      return data.stopset[t] !== 1 && indexable(t);
    });
    var text = ret.bm25.scoreExplained(terms);
    var total = Float64Array.from(text.scores);
    var phonPart = null;
    if (ret.phon && ret.cfg.use_phonetic !== false) {
      var refIdx = ret.ref || ret.bm25;
      var ph = ret.phon.scoreExplained(
        terms,
        function (t) { return refIdx.termIdf(t); },
        function (t) { return refIdx.contains(t); }
      );
      var lam = ret.cfg.lam;
      for (var i = 0; i < total.length; i++) {
        var gate = ret.cfg.confidence_gate === false ? 1.0 : (1.0 - ret.docs[i].conf);
        total[i] = total[i] + lam * gate * ph.scores[i];
      }
      phonPart = ph;
    }
    var order = [];
    for (var k = 0; k < total.length; k++) order.push(k);
    order.sort(function (a, b) { return total[b] - total[a] || a - b; });
    return { terms: terms, textScores: text.scores, textHits: text.hits,
             phon: phonPart, total: total, order: order };
  }

  global.CART = {
    normalize: normalize,
    tokenize: tokenize,
    indexable: indexable,
    phoneSubCost: phoneSubCost,
    weightedEdit: weightedEdit,
    phoneticSimilarity: phoneticSimilarity,
    buildDocuments: buildDocuments,
    buildRetriever: buildRetriever,
    score: score,
    BM25Index: BM25Index,
    PhoneticIndex: PhoneticIndex
  };
})(window);
