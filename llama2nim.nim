#[
Inference for Llama-2 Transformer model in pure Nim. Port of
https://github.com/karpathy/llama2.c

The idea is to keep the no dependency idea alive. However, I use
`cligen` for argument parsing and because it's available its
memfile interface (the latter could be replace by `std/memfiles`, but
initially I thought about using more features)

Example compile: (see README of the original for more details)
$ nim c -d:danger llama2nim.nim

It needs a checkpoint file!

Then run with:
$ ./llama2nim -c <path to checkpoint>
]#

import std / [math, times, random, monotimes]
import cligen / [mfile]

template `%+`[T](x: ptr UncheckedArray[T], i: int): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](cast[uint](x) + (sizeof(T) * i).uint)

template `%+`[T](x: ptr T, i: int): ptr T =
  cast[ptr T](cast[uint](x) + (sizeof(T) * i).uint)

template `%!+`(x: pointer, i: int): pointer =
  ## Different name due to different semantics! Need to multiply by `sizeof`
  ## on the calling side.
  cast[pointer](cast[uint](x) + i.uint)

type
  Buf[T] = object
    size: int
    owned: bool
    data: ptr UncheckedArray[T]

proc `=destroy`[T](x: var Buf[T]) =
  if x.owned and x.data != nil:
    #echo "deallocing: ", x.offsetOf
    dealloc(x.data)

proc `$`*[T](b: Buf[T]): string =
  result = "Buffer(size: " & $b.size & ", owned: " & $b.owned & ", data: " & $b.data.repr & ")"

proc newBuf*[T](size: int): Buf[T] =
  let data = cast[ptr UncheckedArray[T]](alloc(size * sizeof(T)))
  result = Buf[T](owned: true, size: size, data: data)

proc fromPtr*[T](buf: var ptr T, size: int): Buf[T] =
  ## Assigns the region `buf` to `buf + size` to a `Buf[T]` and
  ## increments the input `buf`
  result = Buf[T](owned: false, size: size, data: cast[ptr UncheckedArray[T]](buf))
  buf = buf %+ size

#proc fromPtr*[T](buf: pointer, size: int): Buf[T] =
#  result = Buffer(owned: false, size: size, data: buf)

template getPtr[T](x: Buf[T], idx = 0): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](x.data[idx].addr)

proc getPtr(x: string): ptr UncheckedArray[char] =
  cast[ptr UncheckedArray[char]](x[0].addr)

template `{}`[T](x: Buf[T], idx: int): ptr UncheckedArray[T] = getPtr(x, idx)

proc `[]`[T](b: Buf[T], idx: int): T = b.data[idx]
proc `[]`[T](b: var Buf[T], idx: int): var T = b.data[idx]
proc `[]=`[T](b: var Buf[T], idx: int, val: T) = b.data[idx] = val
proc len[T](b: Buf[T]): int = b.size


# ----------------------------------------------------------------------------
# Transformer and RunState structs, and related memory management

type
  Config = object
    dim: int32 # transformer dimension
    hidden_dim: int32 # for ffn layers
    n_layers: int32 # number of layers
    n_heads: int32 # number of query heads
    n_kv_heads: int32 # number of key/value heads (can be < query heads because of multiquery)
    vocab_size: int32 # vocabulary size, usually 256 (byte-level)
    seq_len: int32 # max sequence length

  TransformerWeights = object
    # token embedding table
    token_embedding_table: Buf[float32]    # (vocab_size, dim)
    # weights for rmsnorms
    rms_att_weight: Buf[float32] # (layer, dim) rmsnorm weights
    rms_ffn_weight: Buf[float32] # (layer, dim)
    # weights for matmuls
    wq: Buf[float32] # (layer, dim, dim)
    wk: Buf[float32] # (layer, dim, dim)
    wv: Buf[float32] # (layer, dim, dim)
    wo: Buf[float32] # (layer, dim, dim)
    # weights for ffn
    w1: Buf[float32] # (layer, hidden_dim, dim)
    w2: Buf[float32] # (layer, dim, hidden_dim)
    w3: Buf[float32] # (layer, hidden_dim, dim)
    # final rmsnorm
    rms_final_weight: Buf[float32] # (dim,)
    # freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Buf[float32] # (seq_len, dim/2)
    freq_cis_imag: Buf[float32] # (seq_len, dim/2)
    # (optional) classifier weights for the logits, on the last layer
    wcls: Buf[float32]

  RunState = object
    # current wave of activations
    x: Buf[float32] # activation at current time stamp (dim,)
    xb: Buf[float32] # same, but inside a residual branch (dim,)
    xb2: Buf[float32] # an additional buffer just for convenience (dim,)
    hb: Buf[float32] # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Buf[float32] # buffer for hidden dimension in the ffn (hidden_dim,)
    q: Buf[float32] # query (dim,)
    k: Buf[float32] # key (dim,)
    v: Buf[float32] # value (dim,)
    att: Buf[float32] # buffer for scores/attention values (n_heads, seq_len)
    logits: Buf[float32] # output logits
    # kv cache
    key_cache: Buf[float32]   # (layer, seq_len, dim)
    value_cache: Buf[float32] # (layer, seq_len, dim)

proc initRunState(cfg: Config): RunState =
  result.x = newBuf[float32](cfg.dim)
  result.xb = newBuf[float32](cfg.dim)
  result.xb2 = newBuf[float32](cfg.dim)
  result.hb = newBuf[float32](cfg.hidden_dim)
  result.hb2 = newBuf[float32](cfg.hidden_dim)
  result.q = newBuf[float32](cfg.dim)
  result.k = newBuf[float32](cfg.dim)
  result.v = newBuf[float32](cfg.dim)
  result.att = newBuf[float32](cfg.n_heads * cfg.seq_len)
  result.logits = newBuf[float32](cfg.vocab_size)
  result.key_cache = newBuf[float32](cfg.n_layers * cfg.seq_len * cfg.dim)
  result.value_cache = newBuf[float32](cfg.n_layers * cfg.seq_len * cfg.dim)

# don't need to call `free`, `=destroy` does it for us :)

# ----------------------------------------------------------------------------
# initialization: read from checkpoint

proc checkpoint_init_weights(w: var TransformerWeights, cfg: Config,
                             f: var ptr float32, shared_weights: int) =
  ## XXX: DECIDE on how to implement. Keep like this? Then replace seq py ptr
  ## Else need to copy! Better this way.
  ## Alternative: use buffer container, e.g. `copyflat`
  w.token_embedding_table = fromPtr[float32](f, cfg.vocab_size * cfg.dim)
  w.rms_att_weight        = fromPtr[float32](f, cfg.n_layers * cfg.dim)
  w.wq                    = fromPtr[float32](f, cfg.n_layers * cfg.dim * cfg.dim)
  w.wk                    = fromPtr[float32](f, cfg.n_layers * cfg.dim * cfg.dim)
  w.wv                    = fromPtr[float32](f, cfg.n_layers * cfg.dim * cfg.dim)
  w.wo                    = fromPtr[float32](f, cfg.n_layers * cfg.dim * cfg.dim)
  w.rms_ffn_weight        = fromPtr[float32](f, cfg.n_layers * cfg.dim)
  w.w1                    = fromPtr[float32](f, cfg.n_layers * cfg.dim * cfg.hidden_dim)
  w.w2                    = fromPtr[float32](f, cfg.n_layers * cfg.hidden_dim * cfg.dim)
  w.w3                    = fromPtr[float32](f, cfg.n_layers * cfg.dim * cfg.hidden_dim)
  w.rms_final_weight      = fromPtr[float32](f, cfg.dim)
  let head_size           = cfg.dim div cfg.n_heads
  w.freq_cis_real         = fromPtr[float32](f, cfg.seq_len * head_size div 2)
  w.freq_cis_imag         = fromPtr[float32](f, cfg.seq_len * head_size div 2)
  w.wcls = if shared_weights > 0: w.token_embedding_table
           else: fromPtr[float32](f, 1)

# ----------------------------------------------------------------------------
# neural net blocks

proc accum[T](a: var Buf[T], b: Buf[T], size: SomeInteger) =
  for i in 0 ..< size:
    a[i] += b[i]

proc rmsnorm[T](o: var Buf[T], x: Buf[T], weight: ptr UncheckedArray[T], size: int) =
  # calculate sum of squares
  assert o.len == x.len
  assert x.len == size
  var ss = T(0.0)
  for j in 0 ..< x.len:
    ss += x[j] * x[j]
  ss /= T(size)
  ss += T(1e-5)
  ss = T(1.0) / sqrt(ss)
  # normalize and scale
  for j in 0 ..< o.len:
    o[j] = weight[j] * (ss * x[j])

proc softmax[T](x: ptr UncheckedArray[T], size: int) =
  # find max value (for numerical stability)
  var max_val = x[0]
  for i in 1 ..< size:
    max_val = max(max_val, x[i])
  # exp and sum
  var sum = T(0.0)
  for i in 0 ..< size:
    x[i] = exp(x[i] - max_val)
    sum += x[i]
  # normalize
  for i in 0 ..< size:
    x[i] /= sum

template softmax[T](x: var Buf[T], size: int) =
  softmax(x.getPtr(), size)

proc matmul[T](xout: var Buf[T], x: Buf[T], w: ptr UncheckedArray[T], n, d: int) =
  # W (d,n) @ x (n,) -> xout (d,)
  #pragma omp parallel for
  for i in `||`(0, d, "parallel for simd"):
    var val = T(0.0)
    for j in 0 ..< n:
      val += w[i * n + j] * x[j]
    xout[i] = val

proc transformer(token, pos: int, cfg: Config, s: var RunState, w: TransformerWeights) =
  # a few convenience variables
  var x = s.x
  let
    dim = cfg.dim
    hidden_dim = cfg.hidden_dim
    head_size = dim div cfg.n_heads ## XXX: div correct, yes?

  # copy the token embedding into x
  copyMem(x.getPtr, w.token_embedding_table{token * dim}, dim * sizeof(float32))

  # pluck out the "pos" row of freq_cis_real and freq_cis_imag
  #echo "Pos = ", pos, " and head size = ", headSize
  let freq_cis_real_row = w.freq_cis_real{pos * head_size div 2}
  let freq_cis_imag_row = w.freq_cis_imag{pos * head_size div 2}

  # forward all the layers
  for l in 0 ..< cfg.n_layers:
    # attention rmsnorm
    rmsnorm(s.xb, x, w.rms_att_weight{l*dim}, dim)

    # qkv matmuls for this position
    matmul(s.q, s.xb, w.wq{l*dim*dim}, dim, dim)
    matmul(s.k, s.xb, w.wk{l*dim*dim}, dim, dim)
    matmul(s.v, s.xb, w.wv{l*dim*dim}, dim, dim)

    # apply RoPE rotation to the q and k vectors for each head
    for h in 0 ..< cfg.n_heads:
      # get the q and k vectors for this head
      let
        q = s.q{h * head_size}
        k = s.k{h * head_size}
      # rotate q and k by the freq_cis_real and freq_cis_imag
      for i in countup(0, head_size, 2):
        let
          q0 = q[i]
          q1 = q[i+1]
          k0 = k[i]
          k1 = k[i+1]
          fcr = freq_cis_real_row[i div 2]
          fci = freq_cis_imag_row[i div 2]
        q[i]   = q0 * fcr - q1 * fci
        q[i+1] = q0 * fci + q1 * fcr
        k[i]   = k0 * fcr - k1 * fci
        k[i+1] = k0 * fci + k1 * fcr

    # save key,value at this time step (pos) to our kv cache
    let
      loff = l * cfg.seq_len * dim # kv cache layer offset for convenience
      key_cache_row = s.key_cache{loff + pos * dim}
      value_cache_row = s.value_cache{loff + pos * dim}
    copyMem(key_cache_row, s.k.getPtr, dim*sizeof(float32))
    copyMem(value_cache_row, s.v.getPtr, dim*sizeof(float32))

    # multihead attention. iterate over all heads
    #pragma omp parallel for
    for h in `||`(0, cfg.n_heads, "parallel for simd"):
      # get the query vector for this head
      let q = s.q{h * head_size}
      # attention scores for this head
      let att = s.att{h * cfg.seq_len}
      # iterate over all timesteps, including the current one
      for t in 0 .. pos: # yes incl pos
        # get the key vector for this head and at this timestep
        let k = s.key_cache{loff + t * dim + h * head_size}
        # calculate the attention score as the dot product of q and k
        var score = 0.0f
        for i in 0 ..< head_size:
          score += q[i] * k[i]
        score /= sqrt(head_size.float32)
        # save the score to the attention buffer
        att[t] = score

      # softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1)

      # weighted sum of the values, store back into xb
      for i in 0 ..< head_size:
        var val = 0.0'f32
        for t in 0 .. pos: # yes incl pos
          val += att[t] * s.value_cache[loff + t * dim + h * head_size + i] # note bad locality
        s.xb[h * head_size + i] = val

    # final matmul to get the output of the attention
    matmul(s.xb2, s.xb, w.wo{l*dim*dim}, dim, dim)

    # residual connection back into x
    accum(x, s.xb2, dim)

    # ffn rmsnorm
    rmsnorm(s.xb, x, w.rms_ffn_weight{l*dim}, dim)

    # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    # first calculate self.w1(x) and self.w3(x)
    matmul(s.hb, s.xb, w.w1{l*dim*hidden_dim}, dim, hidden_dim)
    matmul(s.hb2, s.xb, w.w3{l*dim*hidden_dim}, dim, hidden_dim)

    # F.silu silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    for i in 0 ..< hidden_dim:
      s.hb[i] = s.hb[i] * (1.0'f32 / (1.0'f32 + exp(-s.hb[i])))

    # elementwise multiply with w3(x)
    for i in 0 ..< hidden_dim:
      s.hb[i] = s.hb[i] * s.hb2[i]

    # final matmul to get the output of the ffn
    matmul(s.xb, s.hb, w.w2{l*dim*hidden_dim}, hidden_dim, dim)

    # residual connection
    accum(x, s.xb, dim)

  # final rmsnorm
  rmsnorm(x, x, w.rms_final_weight.getPtr(), dim)

  # classifier into logits
  matmul(s.logits, x, w.wcls.getPtr(), cfg.dim, cfg.vocab_size)

#int sample(float* probabilities, int n) {
proc sample[T](rnd: var Rand, probabilities: Buf[T], n: int): int =
  # sample index from probabilities, they must sum to 1
  let r = T(rnd.rand(1.0))
  var cdf = T(0.0)
  for i in 0 ..< n:
    cdf += probabilities[i]
    if r < cdf:
      return i
  result = n - 1 # in case of rounding errors

proc argmax[T](v: Buf[T], n: int): int =
  # return argmax of v in elements 0..n
  var max_i = 0
  var max_p = v[0]
  for i in 1 ..< n:
    if v[i] > max_p:
      max_i = i
      max_p = v[i]
  result = max_i

proc readVocab(cfg: Config): seq[string] =
  ## Parses the vocabulary file
  result = newSeq[string](cfg.vocab_size)
  var mft = mopen("tokenizer.bin")
  if mft.mem == nil:
    raise newException(IOError, "Unable to open the tokenizer file tokenizer.bin! Run " &
      "python tokenizer.py to convert tokenizer.model -> tokenizer.bin")
  proc readWord(mf: MFile, pos: var int): string =
    ## Reads the (len, cstring) pairs and returns the data in a string.
    ## `pos` is advanced by the necessary bytes
    var l: int32
    # copy the size of the word
    copyMem(l.addr, mft.mem %!+ pos, sizeof(int32))
    inc pos, sizeof(int32)
    result = newString(l)
    # copy the content
    copyMem(result.getPtr, mf.mem %!+ pos, sizeof(char) * l)
    inc pos, l
  var pos = 0
  for i in 0 ..< cfg.vocab_size:
    let wd = readWord(mft, pos)
    result[i] = wd
  mft.close()

proc parseConfigWeights(mf: MFile, file: string): (Config, TransformerWeights) =
  var
    config: Config
    weights: TransformerWeights
  block DataRead:
    if mf.mem == nil:
      raise newException(IOError, "Unable to open the checkpoint file " & $file)
    # read in the config header
    copyMem(config.addr, mf.mem, sizeof(Config))
    echo "Read config: ", config
    # negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = if config.vocab_size > 0: 1 else: 0
    config.vocab_size = abs(config.vocab_size)
    # memory map the Transformer weights into the data pointer
    var weights_ptr = cast[ptr float32](mf.mem %!+ sizeof(Config)) #  div sizeof(float32)))
    checkpoint_init_weights(weights, config, weights_ptr, shared_weights)
  result = (config, weights)

# ----------------------------------------------------------------------------

proc main(checkpoint: string,
          temperature = 0.9'f32, steps = 256): int =
  # seed rng with time. if you want deterministic behavior use temperature 0.0
  var rnd = initRand(now().toTime.toUnix)

  # read in the model.bin file
  let mf = mopen(checkpoint)
  let (config, weights) = parseConfigWeights(mf, checkpoint)

  # right now we cannot run for more than config.seq_len steps
  echo "Model initialized"
  var steps = steps
  if steps <= 0 or steps > config.seq_len:
    steps = config.seq_len

  # read in the tokenizer.bin file
  let vocab = readVocab(config)

  # create and init the application RunState
  var state = initRunState(config)

  # the current position we are in
  let start = getMonoTime()
  var
    next = 0
    token = 1 # 1 = BOS token in Llama-2 sentencepiece
    pos = 0
  echo "<s>" # explicit print the initial BOS token (=1), stylistically symmetric
  while pos < steps:
    # forward the transformer to get logits for the next token
    transformer(token, pos, config, state, weights)

    # sample the next token
    if temperature == 0.0'f32:
      # greedy argmax sampling
      next = argmax(state.logits, config.vocab_size)
    else:
      # apply the temperature to the logits
      for q in 0 ..< config.vocab_size:
        state.logits[q] /= temperature
      # apply softmax to the logits to get the probabilities for next token
      softmax(state.logits, config.vocab_size)
      # we now want to sample from this distribution to get the next token
      next = rnd.sample(state.logits, config.vocab_size)
    stdout.write(vocab[next])
    stdout.flushFile()

    # advance forward
    token = next
    inc pos
  stdout.write("\n")

  # report achieved tok/s
  let stop = getMonoTime()
  echo "achieved tok/s: ", config.seq_len.float / ((stop-start).inMicroSeconds().float / 1e6)

  mf.close()

  # memory and file handles cleanup
  return 0

when isMainModule:
  import cligen
  dispatch(main,
           help = { "checkpoint" : "Input checkpoint file to use",
                    "temperature" : "optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline",
                    "steps" : "number of steps to perform" })
