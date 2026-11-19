# Transformer classroom review — speaker notes

Six slides, approximately 10–12 minutes. The deck is based on the supplied textbook and local Scala implementation. Code was reviewed, not executed.

## Slide 1

Purpose (about 1 minute): Introduce self-attention as a learned way to combine sequence elements. The book contrasts this with recurrent models, which propagate a hidden state through time. A transformer encoder can compute attention across its observed input window without recurrence.

In this implementation, a token is a non-overlapping patch of scalar observations, not a word. Every patch can attend to every observed patch. That is compatible with forecasting because all input patches precede the forecast origin. The target future patch must stay outside the input.

Scope: This is an encoder-only forecaster, with a linear prediction head and no decoder. It is a single-head teaching implementation; a full transformer can use multiple heads and stacked layers. Do not assume all transformers have the same architecture.

Source: John A. Miller, Introduction to Computational Data Science Using ScalaTion, April 9, 2026, §11.10, actual PDF/printed pp. 560–565. The table of contents lists earlier page numbers (536 onward); the deck cites the page numbers on the actual pages. Local code: src/main/scala/scalation/modeling/SimpleTransformer.scala, class declaration at line 298 and encoderForward at line 340. Code reviewed as supplied; no model run or benchmark is claimed.

## Slide 2

Purpose (about 2 minutes): Explain that queries and keys determine relevance, while values carry the information that will be combined. All three come from the same input in self-attention; their projection matrices differ and are learned.

For L patches and model width d, this code has Q, K, V of shape L × d. QKᵀ and the attention probability matrix A have shape L × L. Multiplying A by V returns L × d. Softmax is applied separately to each row; the local AFF.fM implementation uses row-wise matricize. The scaling factor is the square root of the query/key width d_k, not the sequence length. Here d_k = d_mod because there is one head.

The scalar example uses assumed attention weights 0.7 and 0.3 and assumed values 10 and 20 to illustrate a weighted sum. These are not measured model outputs. With vector-valued V, exactly the same weighted combination is applied to every value feature. Ask students which value contributes more and why the result is between 10 and 20.

Distinguish attention weights A (called scr in Scala) from attention output C (scr * v). The method returns both so the backward pass can reuse the probabilities. The displayed code omits the debug print but otherwise preserves the relevant statements.

Sources: textbook §11.10.1, pp. 560–562, especially Eq. 11.59 on p. 561; SimpleTransformer.scala lines 174–181; ActivationFun.scala AFF definition; MatrixD.scala matricize definition.

## Slide 3

Purpose (about 2 minutes): Walk left to right. Position encoding is added first. The first sublayer combines information across patches using attention. The second sublayer applies the same feed-forward network to each patch independently. The residual paths add the sublayer input back before normalization; the equations show exactly what is added. This is post-layer normalization.

encodePositions uses PE(p,2i) = sin(p / 10000^(2i/d_mod)) and PE(p,2i+1) = cos(p / 10000^(2i/d_mod)). The demo has d_mod = 10, so all columns form sin/cos pairs. The current loop leaves an unfilled final column for odd widths. The standalone embedding routines are commented out: raw patch length is the model width in the demo.

The FFN expands each row from 10 to 40 features, applies ReLU, and returns it to 10 features. There are no bias terms or dropout in this implementation. The normalization scale and shift default to fixed scalar values 1 and 0; they are not learned feature vectors here. Multi-head attention would compute several projected attention outputs and combine them, but this code uses only one head.

Implementation detail for technical discussion: layerNorm calls MatrixD.variance, which delegates to VectorD.variance and uses a sample-variance denominator N-1. layerNormBackward uses the usual population-variance derivative form with N. These should be made consistent and checked with finite differences before relying on training. This is a source-review finding, not a training result.

Sources: textbook §§11.10.2–11.10.3, pp. 562–564, including Fig. 11.9; SimpleTransformer.scala encodePositions at line 127, layerNorm at 225, layerNormBackward at 239, ffn at 261, encoderForward at 340; MatrixD.scala variance at 1696 and VectorD.scala variance at 1375.

## Slide 4

Purpose (about 1.5 minutes): Tie every dimension to the supplied runnable example. patchLength = 10, lookPatches = 4, forePatches = 1. Thus d_mod = 10, d_ff = 40, and horizon = 10. A forecast predicts ten scalar time steps directly, not one scalar or one token ID.

The first illustrated window consumes observations 1–40 and targets 41–50; the next window shifts one patch, consuming 11–50 and targeting 51–60. These are illustrative one-based indices, not real lake-level values. Neighboring examples overlap even though the patches themselves do not.

patchify creates floor(m / patchLength) full patches and drops any incomplete trailing patch. createSlidingWindowDataset produces P - lookback - forePatches + 1 examples when there are enough patches. Targets are flattened into a VectorD of length forePatches * patchLength.

The attention matrix is 4 × 4 because tokens are patches, not individual scalar observations. The FFN expands feature width without changing patch count. The final encoder output is flattened to length 40, then multiplied by wP of shape 40 × 10 to produce a horizon-length vector.

Source: SimpleTransformer.scala patchify lines 43–49, createSlidingWindowDataset lines 88–120, demo configuration and data preparation lines 476–499. Units of lake observations are deliberately not asserted here; the slide uses time steps.

## Slide 5

Purpose (about 2 minutes): Map the conceptual block to the actual forward-pass statements. The line x += posEnc executes immediately before this excerpt and is discussed on slide 6. In ScalaTion, flattened *: wP denotes the vector–matrix multiplication that produces the forecast. The method returns EncoderCache, retaining intermediate tensors for manual backpropagation.

The training loop processes one training example at a time in sequence. It computes error = pred - target and MSE = (error dot error) / target.dim. The derivative seed is d_pred = error * (2.0 / target.dim). encoderBackward propagates this through the prediction head, both normalization layers, FFN, attention, and Q/K/V projections. Residual connections require adding gradients from both paths.

The six trainable matrices updated after every instance are wQ, wK, wV, w1, w2, and wP. This is incremental gradient descent, not a mini-batch optimizer or Adam. The supplied demo sets learningRate = 0.007 and totalEpochs = 40. These are example settings, not tuned recommendations or evidence of convergence.

Code-review status: The supplied source was inspected, but the training program was not executed for this presentation. No training loss, test accuracy, or forecast plot is invented. The final sample trace in the source is from training data, as explained on the next slide.

Sources: SimpleTransformer.scala encoderForward lines 340–357, encoderBackward lines 364–405, train lines 416–454, demo settings lines 501–504.

## Slide 6

Purpose (about 2 minutes): Ask the questions before revealing the answers verbally.

Answers:
1. Scaling by sqrt(d_k) counteracts the tendency of dot-product magnitudes to grow with feature width, helping avoid excessively sharp softmax probabilities and tiny gradients. It does not itself normalize the attention rows; softmax does that.
2. Position encodings supply location information. Content-only self-attention is permutation-equivariant; it cannot itself know which position a patch occupies. The subsequent fixed flattening head can depend on position, but that is distinct from giving attention position information.
3. Four input patches create four queries and four keys. Thus QKᵀ is 4 × 4 even though each patch has ten features and the forecast contains ten scalar observations.

Code caveats, from static inspection:
• encoderForward uses x += posEnc (line 341). MatrixD.+= mutates the matrix. train repeatedly passes the same matrices from trainingData, so position encodings accumulate over epochs; the later sample trace adds them again. A repair would use a fresh local sum and consistently use it in forward calculations and the cache. No source changes were made for this deck.
• The sampleInput = xTrain.head and yy = yTrain.head evaluation uses a training example. It is not held-out validation, despite its printed label. For a performance study, use chronological training/validation/test splits, ensure training targets precede the split, fit any preprocessing on training data, and compare with appropriate simple forecasting baselines.
• layerNorm computes sample variance through MatrixD.variance → VectorD.variance (denominator N-1), while layerNormBackward uses the usual N-based normalization derivative. Align the forward and backward convention and verify gradients numerically before relying on optimization results. No numerical gradient check was run for this slide deck.

Reference list:
John A. Miller. Introduction to Computational Data Science Using ScalaTion. April 9, 2026. Supplied file comp_data_science (1).pdf, §11.10 and Fig. 11.9, actual pp. 560–565; related attention discussion §11.9.3 on pp. 558–559.
Local SimpleTransformer.scala, John A. Miller, header dated November 12, 2026; src/main/scala/scalation/modeling/SimpleTransformer.scala. Supporting implementation checks: MatrixD.scala and VectorD.scala in scalation/mathstat, plus ActivationFun.scala. This later local implementation supplements the book; it is not claimed to be printed in the April PDF.

Suggested closing question: Why can an unmasked encoder still be used for forecasting here? Answer: it attends only over the observed historical window; future targets are not part of the input.
