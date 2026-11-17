
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John A. Miller
 *  @version 2.0
 *  @date    Thu Nov 12 14:09:11 EST 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Simple Implementation for an Encoder-Only Transformer, e.g., used for
 *           Time Series Forecasting (TSF)
 *
 *  @see sebastianraschka.com/blog/2023/self-attention-from-scratch.html
 *  @see arxiv.org/pdf/1706.03762.pdf (main paper)
 *
 *  @note AI Assisted Code
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scala.collection.mutable.ArrayBuffer
import scala.math.{cos, max, sin, sqrt}

import scalation.mathstat._
import scalation.modeling.ActivationFun.f_softmax
import scalation.random.RandomMatD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleTransformer` object implements utility methods for transformers.
 */
object SimpleTransformer:

    val eps = 1E-5                                            // very small value

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Patchify the univariate time series y by breaking it into non-overlapping
     *  patches of length pl.  This simple implementation assumes stride s = pl,
     *  but PatchTST uses pl = 16 and s = 8 as defaults.
     *  @param y   the given univariate time series
     *  @param pl  the patch length
     */
    def patchify (y: VectorD, pl: Int): MatrixD =
        val m  = y.dim
        val np = m / pl
        val x  = new MatrixD (np, pl)
        for i <- x.indices do x(i) = y(i*pl until (i+1)*pl)
        x
    end patchify

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prepare sliding window training samples from a patchified time series matrix.
     *  @param patchedMatrix  The output matrix from the patchify method (num_patches x patch_length)
     *  @param lookback       How many patches the transformer looks at (seq_len)
     *  @return (Seq [InputMatrix], Seq [TargetMatrix])
     *
    def createSlidingWindowDataset (patchedMatrix: MatrixD, lookback: Int): (Seq [MatrixD], Seq [MatrixD]) =
        val trainingInputs  = ArrayBuffer [MatrixD] ()
        val trainingTargets = ArrayBuffer [MatrixD] ()

        // Slide across the total available patches
        for i <- 0 until (patchedMatrix.dim - lookback) do
            // Input sequence matrix of shape (lookback x patch_length)
            val xInput = new MatrixD (lookback, patchedMatrix.dim2)
            for row <- 0 until lookback do
                xInput(row) = patchedMatrix(i + row)
        
            // Target matrix (the very next patch) of shape (1 x patch_length)
            val yTarget = new MatrixD (1, patchedMatrix.dim2)
            yTarget(0)  = patchedMatrix(i + lookback)

            trainingInputs  += xInput
            trainingTargets += yTarget

        (trainingInputs.toSeq, trainingTargets.toSeq)
    end createSlidingWindowDataset
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prepare sliding window training samples from a patchified time series matrix.
     *  Each target sample is returned as a plain VectorD representing true future values,
     *  making it perfectly compatible with a linear prediction head.
     *  @param patchedMatrix  the output matrix from the patchify method (num_patches x patch_length)
     *  @param lookback       how many historical patches the transformer looks at (seq_len)
     *  @param forePatches    how many patches into the future you want to predict (typically 1)
     *  @return (Seq [MatrixD], Seq [VectorD])
     */
    def createSlidingWindowDataset (patchedMatrix: MatrixD, lookback: Int, forePatches: Int = 1):
            (Seq [MatrixD], Seq [VectorD]) =
        
        val trainingInputs  = ArrayBuffer [MatrixD] ()
        val trainingTargets = ArrayBuffer [VectorD] ()

        // Calculate the maximum start index to avoid running out of available future patches
        val maxStartIndex = patchedMatrix.dim - lookback - forePatches

        for i <- 0 to maxStartIndex do
            // Construct the Input History Matrix of shape (lookback x patch_length)
            val xInput = new MatrixD (lookback, patchedMatrix.dim2)
            for row <- 0 until lookback do
                xInput(row) = patchedMatrix(i + row)
            
            // Construct the Target Future Values Vector by flattening the target patches
            // If foretPatches = 1, it takes exactly the next 1 patch vector of size patch_length
            val targetLen  = forePatches * patchedMatrix.dim2
            val yTargetVec = new VectorD (targetLen)
            
            var idx = 0
            for p <- 0 until forePatches do
                val nextPatchRow = patchedMatrix(i + lookback + p)
                for element <- 0 until nextPatchRow.dim do
                    yTargetVec(idx) = nextPatchRow(element)
                    idx += 1
            
            trainingInputs  += xInput
            trainingTargets += yTargetVec
        end for

        (trainingInputs.toSeq, trainingTargets.toSeq)
    end createSlidingWindowDataset

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Encode all the positions in the time series as vectors of length d_mod.
     *  @param len    the sequence length
     *  @param d_mod  the dimensionality of the model (d_mod = d_k here)
     */
    def encodePositions (len: Int, d_mod: Int): MatrixD =
        val pe = new MatrixD (len, d_mod)
        for k <- pe.indices do
            for i <- 0 until d_mod/2 do
                val den = 10000.0~^(2.0*i/d_mod)
                pe(k, 2*i)   = sin (k / den)
                pe(k, 2*i+1) = cos (k / den)
        pe
    end encodePositions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use a matrix transformation containing learnable weights to embed each patch
     *  vector into a higher dimensional space (providing enhanced vector similarity).
     *  The dimensionality of the embedding space is d_mod.  For this simple
     *  implementation d_mod = d_k as there is only one attention head.
     *  @note:  currently not used (not needed for small univariate times series)
     *  @param xx  the matrix containing each patch as a row
     *  @param wE  the dimensionality of the embedding space
     *
    def embed (xx: MatrixD, wE: MatrixD): MatrixD = xx * wE
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backpropagation through the linear embedding layer.
     *  @note:  currently not used (not needed for small univariate times series)
     *  @param xx      the original input matrix to embed
     *  @param wE      the embedding weight matrix
     *  @param d_out   incoming gradient matrix of shape (np x d_mod)
     *  @return (d_xx, d_wE) gradients w.r.t the input patches and weights
     *
    def embedBackward (xx: MatrixD, wE: MatrixD, d_out: MatrixD): (MatrixD, MatrixD) =
        val d_xx = d_out * wE.ᵀ
        val d_wE = xx.ᵀ * d_out
        (d_xx, d_wE)
    end embedBackward
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Based on the Query (Q), Key (K), and Value (V) matrices, compute the attention.
     *
     *      att = softmax (QK^T/√d_k) V
     *
     *  @param q    the Query: the input of interest
     *  @param k    the Key: other locations to compare it with (for similarity)
     *  @param v    the Value: the input value at the key locations
     *  @param d_k  the dimensionality of Query, Key, and Value (if different use d_v)
     */
    def attention (q: MatrixD, k: MatrixD, v: MatrixD, d_k: Int): (MatrixD, MatrixD) =
        val qkt = q * k.ᵀ                                     // repeated dot product
        val sdp = qkt / sqrt (d_k)                            // scaled dot product (sdp)
        val scr = f_softmax.fM (sdp)                          // attention scores
        println (s"attention: qkt = $qkt, sdp = $sdp, scr = $scr")
        (scr, scr * v)                                        // score, attention (Q, K, V)
    end attention

   //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backpropagation through the Scaled Dot-Product Attention layer.
     *  @param q        the Query matrix
     *  @param k        the Key matrix
     *  @param v        the Value matrix
     *  @param scr      the saved attention scores matrix from the forward pass
     *  @param d_k      the dimensionality dimension parameter
     *  @param d_out    incoming gradient from the next upper layer
     *  @return (dQ, dK, dV) matrix gradients
     */
    def attentionBackward (q: MatrixD, k: MatrixD, v: MatrixD, scr: MatrixD,
                           d_k: Int, d_out: MatrixD): (MatrixD, MatrixD, MatrixD) =
        val dV = scr.ᵀ * d_out                        // gradient w.r.t. Values (V)
        val d_scr = d_out * v.ᵀ                       // gradient w.r.t. Attention Score probabilities (scr)

        // Gradient through Row-wise Softmax
        val d_sdp = new MatrixD (scr.dim, scr.dim2)
        for i <- scr.indices do
            val s_row  = scr(i)
            val ds_row = d_scr(i)
            val dot    = s_row dot ds_row
            for j <- 0 until scr.dim2 do
                d_sdp(i, j) = s_row(j) * (ds_row(j) - dot)

        val d_qkt = d_sdp / sqrt (d_k)                // gradient through scaling factor

        // Gradients with respect to Queries (Q) and Keys (K)
        val dQ = d_qkt * k
        val dK = d_qkt.ᵀ * q

        (dQ, dK, dV)
    end attentionBackward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform layer normalization on matrix x.  While batch normalization normalizes
     *  over a mini-batch, layer normalization normalizes over a single instance or row.
     *  @see www.geeksforgeeks.org/deep-learning/what-is-layer-normalization/
     *  An affine transformation is supported via the γ and β given/learnable parameters.
     *  Option:  val xn = ((xt - xt.mean) / (xt.stdev + eps)).ᵀ
     *  @param x  the matrix to normalize
     *  @param γ  the scaling given/learnable parameter (defaults to 1.0)
     *  @param β  the shifting given/learnable parameter (defaults to 0.0)
     */
    def layerNorm (x: MatrixD, γ: Double = 1.0, β: Double = 0.0): MatrixD =
        val xt = x.ᵀ                                                   // transpose since ScalaTion computes means and stdevs column-wise
        val xn = ((xt - xt.mean) / (xt.variance + eps).sqrt).ᵀ         // z-transform to mean zero and stdev one
        xn * γ + β                                                     // allows scaling and shifting for flexibility
    end layerNorm

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backpropagation through the Layer Normalization matrix computation.
     *  Caveat: currently only works with γ and β locked (not learned).
     *  @param x      original input matrix before normalization
     *  @param d_out  incoming gradient matrix
     *  @param γ      the scaling given/learnable parameter (defaults to 1.0)
     *  @return d_x gradients (note:  dγ, dβ would also be returned if γ and β are learned)
     */
    def layerNormBackward (x: MatrixD, d_out: MatrixD, γ: Double = 1.0): MatrixD =
        val N       = x.dim2.toDouble                                  // number of features per row
        val xt      = x.ᵀ                                              // x transpose
        val std_inv = (xt.variance + eps).map (1.0 / sqrt (_))         // 1/standard deviation
        val x_norm  = ((xt - xt.mean) / (xt.variance + eps).sqrt).ᵀ    // x normalized
        val d_xnorm = d_out * γ                                        // scale upstream gradient by γ

        val d_x = new MatrixD (x.dim, x.dim2)                          // row-by-row gradients with respect to input x
        for i <- x.indices do                                          // for each row/instance
            val dxn = d_xnorm(i)                                                                     
            val xn  = x_norm(i)
            d_x(i) = (dxn - dxn.sum / N - xn * ((dxn dot xn) / N)) * std_inv(i)
        d_x
    end layerNormBackward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Position-wise Feed-Forward Network forward step.
     *  @param x   Input matrix from attention layer norm
     *  @param w1  Weights of the first linear projection (d_mod x d_ff)
     *  @param w2  Weights of the second linear projection (d_ff x d_mod)
     *  @return (final_output, intermediate_hidden_before_relu)
     */
    def ffn (x: MatrixD, w1: MatrixD, w2: MatrixD): (MatrixD, MatrixD) =
        val z_linear = x * w1
        val z_act    = ActivationFun.f_reLU.fM (z_linear)    // apply element-wise ReLU activation
        (z_act * w2, z_linear)
    end ffn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Position-wise Feed-Forward Network backward step.
     */
    def ffnBackward (x: MatrixD, w1: MatrixD, w2: MatrixD, z_linear: MatrixD, z_act: MatrixD,
                     d_out: MatrixD): (MatrixD, MatrixD, MatrixD) =
        val d_w2    = z_act.ᵀ * d_out                        // gradients w.r.t second weight matrix (W2)
        val d_z_act = d_out * w2.ᵀ                           // gradient flowing back into the activation layer
        
        // Backprop through element-wise ReLU derivative
        val d_z_linear = new MatrixD (z_linear.dim, z_linear.dim2)
        for i <- z_linear.indices; j <- z_linear.indices2 do
            d_z_linear(i, j) = if z_linear(i, j) > 0.0 then d_z_act(i, j) else 0.0
            
        // Gradients w.r.t first weight matrix (W1) and input layer (X)
        val d_w1 = x.ᵀ * d_z_linear
        val d_x  = d_z_linear * w1.ᵀ
        
        (d_x, d_w1, d_w2)
    end ffnBackward

end SimpleTransformer

import SimpleTransformer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleTransformer` class provides a stateful, object-oriented implementation
 *  of an Encoder-only Transformer Layer.
 *  @param d_mod     the embedding dimension size of input patches (model dimensionality)
 *  @param d_ff      the internal hidden layer dimension for the Feed-Forward Network
 *  @param lookback  the length of the lookback window (how far into the past)
 *  @param horizon   the forecasting horizon (how far in the future)
 */
class SimpleTransformer (val d_mod: Int, val d_ff: Int, val lookback: Int, val horizon: Int):

    private val posEnc = encodePositions (lookback, d_mod)               // positional encoding

    // Simple Learnable Parameter Weights Matrices Initialization
    // In production, initialize these using randomized distributions (e.g., Xavier/Glorot)

    var wQ = randomMatrix (d_mod, d_mod, stream = 1)                     // Query Weights Matrix
    var wK = randomMatrix (d_mod, d_mod, stream = 2)                     // Key Weights Matrix
    var wV = randomMatrix (d_mod, d_mod, stream = 3)                     // Value Weights Matrix
    var w1 = randomMatrix (d_mod, d_ff,  stream = 4)                     // FFN First Projection Layer
    var w2 = randomMatrix (d_ff,  d_mod, stream = 5)                     // FFN Second Projection Layer
    var wP = randomMatrix (lookback * d_mod, horizon, stream = 6)        // maps flattened matrix to horizon (prediction head)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Helper method leveraging ScalaTion's `RandomMatD` using different streams
     *  to keep the randomized weights unique for each matrix layer.
     */
    private def randomMatrix (dim: Int, dim2: Int, stream: Int): MatrixD =
        RandomMatD (dim, dim2, max = 0.1, min = -0.1, density = 1.0, stream = stream).gen
    end randomMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `EncoderCache` case class serves as a container for state information computed
     *  during forward propogation.
     */
    case class EncoderCache (x: MatrixD, q: MatrixD, k: MatrixD, v: MatrixD, scr: MatrixD, 
                             attn_out: MatrixD, x_norm1: MatrixD, h_linear: MatrixD, 
                             ffn_out: MatrixD, out: MatrixD, flattened: VectorD, pred: VectorD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `EncoderCache` case class serves as a container for gradients computed
     *  during backward propagation.
     */
    case class EncoderGradients (d_x: MatrixD, d_wQ: MatrixD, d_wK: MatrixD, d_wV: MatrixD, 
                                 d_w1: MatrixD, d_w2: MatrixD, d_wP: MatrixD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Full Encoder Block Forward Pipeline to make prediction pred and record intermediates.
     *  @param x  the input sequence matrix of historical patches (lookback x d_mod)
     */
    def encoderForward (x: MatrixD): EncoderCache =
        x += posEnc                                      // inject patch order information (positional encoding)

        // Sub-Layer 1: Self-Attention
        val (q, k, v) = (x * wQ, x * wK, x * wV)         // Query, Key, Value
        val (scr, attn_out) = attention (q, k, v, d_mod)
        val x_norm1 = layerNorm (x + attn_out)
        
        // Sub-Layer 2: Feed-Forward Network
        val (ffn_out, h_linear) = ffn (x_norm1, w1, w2)
        val out = layerNorm (x_norm1 + ffn_out)

        // Forward pass through Prediction Head
        val flattened = out.flatten                      // flatten transformer output matrix into a 1D Vector
        val pred = flattened *: wP                       // linear projection: VectorD * MatrixD = VectorD of size horizon
        
        EncoderCache (x, q, k, v, scr, attn_out, x_norm1, h_linear, ffn_out, out, flattened, pred)
    end encoderForward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Full Encoder Block Backward Pass to compute parameter gradients.
     *  @param cache   the intermediate forward-pass matrices saved during `encoderForward`
     *  @param d_pred  the loss gradient with respect to the forecast vector (length = horizon)
     */
    def encoderBackward (cache: EncoderCache, d_pred: VectorD): EncoderGradients =

        // Backward pass through Prediction Head
        // Gradient w.r.t the prediction weight matrix (wP)
        // dL/dwP = flattened^T (as col vector) * d_pred (as row vector)
        val d_wP = MatrixD.outer(cache.flattened, d_pred)
        
        // Gradient backpropagating to the flattened vector
        // dL/dFlattened = d_pred * wP^T
        val d_flattened = wP * d_pred // Returns a VectorD of size (lookback * d_mod)
        
        // Reshape d_flattened back into a MatrixD of shape (lookback x d_mod)
        val d_out = new MatrixD(lookback, d_mod)
        for i <- 0 until lookback do
            d_out(i) = d_flattened(i * d_mod until (i + 1) * d_mod)

        // Backward through Sub-Layer 2 LayerNorm
        val d_sub2_in = layerNormBackward (cache.x_norm1 + cache.ffn_out, d_out)
        
        // Backward through FFN
        val h_activated = new MatrixD(cache.h_linear.dim, cache.h_linear.dim2)
        for i <- cache.h_linear.indices; j <- 0 until cache.h_linear.dim2 do 
            h_activated(i, j) = max (0.0, cache.h_linear(i, j))
        
        val (d_x_norm1_ffn, d_w1, d_w2) = ffnBackward (cache.x_norm1, w1, w2, cache.h_linear, h_activated, d_sub2_in)
        val d_x_norm1 = d_sub2_in + d_x_norm1_ffn 
        
        // Backward through Sub-Layer 1 LayerNorm
        val d_sub1_in = layerNormBackward (cache.x + cache.attn_out, d_x_norm1)
        
        // Backward through Attention
        val (d_q, d_k, d_v) = attentionBackward (cache.q, cache.k, cache.v, cache.scr, d_mod, d_sub1_in)
        
        // Compute gradients w.r.t our learnable matrices
        val d_wQ = cache.x.ᵀ * d_q
        val d_wK = cache.x.ᵀ * d_k
        val d_wV = cache.x.ᵀ * d_v
        
        // Final gradient tracking out of the layer input 
        val d_x = d_sub1_in + (d_q * wQ.ᵀ) + (d_k * wK.ᵀ) + (d_v * wV.ᵀ)
        
        EncoderGradients (d_x, d_wQ, d_wK, d_wV, d_w1, d_w2, d_wP)
    end encoderBackward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `train` method provides a Training Optimization Loop using Incremental Gradient Descent (IGD).
     *  It updates parameters immediately after evaluating a single instance.
     *  @param trainingData   Sequence of input training matrices, each shape (seq_len x d_mod)
     *  @param targetVectors  Sequence of target label matrices, each shape (seq_len x d_mod)
     *  @param lr             Learning Rate scaling step size (e.g., 0.01)
     *  @param epochs         Total training iterations over the dataset
     */
    def train (trainingData: Seq [MatrixD], targetVectors: Seq [VectorD], lr: Double, epochs: Int): Unit =

        println(s"Starting Incremental Gradient Descent for students across $epochs epochs...")
        
        for epoch <- 1 to epochs do
            var epochLoss = 0.0
            
            // Loop through each individual input-target instance pair sequentially
            for i <- trainingData.indices do
                val x = trainingData(i)
                val target = targetVectors(i)
                
                // Forward Pass (Evaluate single instance)
                val cache = encoderForward (x)
                
                // Explicit Vector-based MSE Loss
                val error  = cache.pred - target
                epochLoss += (error dot error) / target.dim
                
                // Gradient seed at the prediction head output
                val d_pred = error * (2.0 / target.dim)
                
                // Backward Pass (Propagate gradients through network) ---
                val grads = encoderBackward (cache, d_pred)
                
                // Incremental Weight Update Step ---
                // Weights are altered immediately using current instance gradients
                wQ -= grads.d_wQ * lr
                wK -= grads.d_wK * lr
                wV -= grads.d_wV * lr
                w1 -= grads.d_w1 * lr
                w2 -= grads.d_w2 * lr
                wP -= grads.d_wP * lr
            end for
            
            // Periodically report average performance metrics
            if epoch % max (1, epochs / 5) == 0 || epoch == 1 then
                val avgLoss = epochLoss / trainingData.size
                println (s"Epoch $epoch completed -> Average Dataset Loss: $avgLoss")
        end for
        println ("Training optimization loop complete.")
    end train

end SimpleTransformer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleTransformerTest` main method tests the `SimpleTransformer` class
 *  on the univariate Lake-Level dataset.
 *  > runMain scalation.modeling.forecasting.neuralforecasting.simpleTransformerTest
 */
@main def simpleTransformerTest (): Unit =

    banner ("Initializing Transformer Test on Example_LakeLevels Dataset")

    // Retrieve the single univariate raw vector from ScalaTion's example module
    val y: VectorD = Example_LakeLevels.y
    println (s"Loaded raw dataset size: ${y.dim} time-steps.")

    // Define structural Hyperparameters
    val patchLength = 10                             // column size per patch (d_mod size)
    val lookPatches = 4                              // how many history patches to look back at (seq_len)
    val forePatches = 1                              // how many future patches to predict ahead

    val d_mod   = patchLength
    val d_ff    = patchLength * 4                    // standard expansion ratio (d_mod * 4)
    val horizon = forePatches * patchLength          // forecasting horizon

    // Step through the Data Preparation Pipeline
    println ("\n[Step 1/3] Converting raw vector array to matrix patches...")
    val patchedMat = patchify (y, patchLength)
    println (s"  Patched matrix dimensions: (${patchedMat.dim} patches x ${patchedMat.dim2} dimensions)")

    println ("\n[Step 2/3] Chunking patches into sliding window training records...")
    val (xTrain, yTrain) = createSlidingWindowDataset (patchedMat, lookback = lookPatches, forePatches = forePatches)

    println(s"  Total generated training sequence instances: ${xTrain.size}")
    println(s"  Single Input Sequence Matrix Shape: (${xTrain.head.dim} x ${xTrain.head.dim2})")
    println(s"  Single Target Label Vector Dimensions: ${yTrain.head.dim}")

    // Instantiate the Stateful Object-Oriented Network Architecture
    println ("\n[Step 3/3] Instantiating SimpleTransformer and breaking matrix symmetry...")
    val transformer = new SimpleTransformer (d_mod = d_mod, d_ff = d_ff, lookback = lookPatches, horizon = horizon)

    // Run the Incremental Online Training Optimization Loop
    val learningRate = 0.007
    val totalEpochs  = 40
    
    transformer.train (trainingData = xTrain, targetVectors = yTrain, lr = learningRate, epochs = totalEpochs)

    // Visual inference verification check and standard metrics
    banner ("Post-Training Validation Sample Trace")
    val sampleInput = xTrain.head
    val yy          = yTrain.head
    val finalCache  = transformer.encoderForward (sampleInput)
    val yp          = finalCache.pred

    println (s"True Next Values (Target Vector) :\n$yy")
    println (s"Model Forecasted Output Vector   :\n$yp")
    println ("==========================================================")

    object ft extends Fit (1, yy.dim)
    ft.diagnose (yy, yp)
    val qof = ft.fit
    println (FitM.fitMap (qof, qoF_names))

end simpleTransformerTest

