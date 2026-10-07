package scalation
package modeling

import java.io.PrintWriter
import java.nio.file.{Files, Paths}
import scala.collection.mutable.{ArrayBuffer, LinkedHashSet => LSET}
import scala.util.Random
import scalation.mathstat._

// One outer split is reserved BEFORE feature selection. All fitting, scaling and
// numerical-rank screening use only its training rows. No validate() calls occur
// during selection or scoring. Run: runMain scalation.modeling.project2SymbolicRegression
@main def project2SymbolicRegression (): Unit =
    for key <- Array ("auto_mpg", "concrete", "airfoil") do runProject2SymbolicDataset (key)
end project2SymbolicRegression

// Shared by the three dataset-specific entry points, preventing divergent results.
def runProject2SymbolicDataset (key: String): Unit =
    val (powers, cross, cross3) = key match
        case "auto_mpg" => (LSET (-2.0, -1.0, 0.5, 2.0), false, false)
        case "concrete" => (LSET (1.0, 0.5, 2.0), false, false)
        case "airfoil"  => (LSET (1.0, 0.0, 0.5, 2.0), true, true)
        case _ => throw new IllegalArgumentException (s"Unknown dataset: $key")
    val out = "project2/results/symbolic/"
    Files.createDirectories (Paths.get (out))
    val (xy, names) = MatrixD.loadH (s"data/$key.csv", fullPath = true)
    val p = xy.dim2 - 1
    val x = xy(?, 0 until p)
    val y = xy(?, p)
    val nTest = (y.dim * 0.2).toInt
    val shuffled = new Random (42).shuffle ((0 until y.dim).toList)
    val trainIdx = shuffled.take (y.dim - nTest).toIndexedSeq
    val testIdx = shuffled.drop (y.dim - nTest).toIndexedSeq
    require (trainIdx.toSet.intersect (testIdx.toSet).isEmpty)
    val yTr = y(trainIdx)
    val yTe = y(testIdx)
    val custom = if key == "airfoil" then Seq (Array ((2, 1.0), (0, -1.0), (4, -1.5)))
                 else Seq.empty [Array [Xj2p]]
    val (basis, termNames) = SymbolicRegression.buildMatrix (
        x, names.take (p), powers, null, true, cross, cross3, custom*)
    val means = new VectorD (basis.dim2)
    val scales = VectorD.one (basis.dim2)
    val scaled = basis.copy
    val keep = LSET (0)
    val orthogonal = ArrayBuffer (VectorD.one (trainIdx.size) / math.sqrt (trainIdx.size.toDouble))
    val rejected = ArrayBuffer [String] ()
    for j <- 1 until basis.dim2 do
        val v = basis(trainIdx)(?, j)
        val mu = v.mean
        val sd = math.sqrt ((v - mu).normSq / v.dim)
        means(j) = mu
        scales(j) = sd
        if sd > 0 && sd.isFinite then
            scaled(?, j) = (basis(?, j) - mu) / sd
            val z = scaled(trainIdx)(?, j)
            var residual = z.copy
            // Reorthogonalize twice to screen numerical redundancy reliably.
            for _ <- 0 until 2 do
                for q <- orthogonal do residual = residual - q * (q dot residual)
            if residual.norm > 1e-6 * z.norm then
                keep += j
                orthogonal += residual / residual.norm
            else rejected += termNames(j)
        else rejected += termNames(j)
    val columns = keep.toArray
    val zTr = scaled(trainIdx)(?, keep)
    val zTe = scaled(testIdx)(?, keep)
    val keptNames = columns.map (termNames(_))
    val previous = FeatureSelection.fullset_FS
    // The complete matrix passed here is already restricted to OUTER TRAINING rows.
    val (selected, path) = try
        FeatureSelection.fullset_FS = true
        val mod = new SymbolicRegression (zTr, yTr, keptNames, powers, null)
        val (_, selectionPath) = mod.forwardSelAll ("none")(using QoF.smapeC.ordinal)
        (mod.getBest.mod_cols, selectionPath)
    finally FeatureSelection.fullset_FS = previous
    val selectedOriginal = selected.toArray.map (columns(_))
    val fit = new Regression (zTr(?, selected), yTr, selectedOriginal.map (termNames(_)))
    fit.train ()
    val ypTr = fit.predict (zTr(?, selected))
    val ypTe = fit.predict (zTe(?, selected))
    require (ypTe.indices.forall (i => ypTe(i).isFinite), s"Nonfinite predictions: $key")
    val baseline = new Regression (VectorD.one (trainIdx.size) +^: x(trainIdx), yTr)
    baseline.train ()
    val baseTe = baseline.predict (VectorD.one (testIdx.size) +^: x(testIdx))
    def metrics (actual: VectorD, prediction: VectorD): (Double, Double, Double) =
        val e = actual - prediction
        (1 - e.normSq / (actual - actual.mean).normSq,
         math.sqrt (e.normSq / actual.dim), e.map (math.abs).mean)
    val metricOut = new PrintWriter (s"$out${key}_metrics.csv")
    metricOut.println ("model,split,rows,terms,r2,rmse,mae,test_rows_in_selection")
    for (label, split, actual, predicted, k) <- Seq (
        ("selected", "train", yTr, ypTr, selected.size - 1),
        ("selected", "test", yTe, ypTe, selected.size - 1),
        ("linear_baseline", "test", yTe, baseTe, p)) do
        val (r2, rmse, mae) = metrics (actual, predicted)
        metricOut.println (s"$label,$split,${actual.dim},$k,$r2,$rmse,$mae,0")
    metricOut.close ()
    val coefficients = new PrintWriter (s"$out${key}_coefficients.csv")
    coefficients.println ("term,coefficient,training_mean,training_scale")
    for (j, k) <- selectedOriginal.zipWithIndex do
        coefficients.println (s"${termNames(j)},${fit.parameter(k)},${means(j)},${scales(j)}")
    coefficients.close ()
    val predictions = new PrintWriter (s"$out${key}_test_predictions.csv")
    predictions.println ("row_index,y,prediction,linear_baseline")
    for i <- testIdx.indices do predictions.println (s"${testIdx(i)},${yTe(i)},${ypTe(i)},${baseTe(i)}")
    predictions.close ()
    val splitOut = new PrintWriter (s"$out${key}_split.csv")
    splitOut.println ("row_index,split")
    for i <- trainIdx do splitOut.println (s"$i,train")
    for i <- testIdx do splitOut.println (s"$i,test")
    splitOut.close ()
    val pathOut = new PrintWriter (s"$out${key}_selection_path.csv")
    pathOut.println ("step,r2_percent,adjusted_r2_percent,smape_percent")
    for (r, i) <- path.zipWithIndex do pathOut.println (s"$i,${r(0)},${r(1)},${r(2)}")
    pathOut.close ()
    val summary = s"Response: ${names(p)}\nTraining rows: ${trainIdx.size}; test rows: ${testIdx.size}\n" +
        s"Candidate terms: ${basis.dim2 - 1}; after training rank screening: ${keep.size - 1}\n" +
        s"Rejected terms: ${rejected.mkString (", ")}\n" +
        s"Selected terms: ${selectedOriginal.map (termNames(_)).mkString (", ")}\n" +
        s"Training metrics (R2, RMSE, MAE): ${metrics (yTr, ypTr)}\n" +
        s"Test metrics (R2, RMSE, MAE): ${metrics (yTe, ypTe)}\n" +
        "Test rows used in selection: 0\n"
    Files.writeString (Paths.get (s"$out${key}_summary.txt"), summary)
    println (summary)
end runProject2SymbolicDataset
