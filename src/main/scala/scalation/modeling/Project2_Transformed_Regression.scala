package scalation
package modeling

import java.io.PrintWriter
import java.nio.file.{Files, Paths}
import scala.util.Random

import scalation.mathstat._

// To execute: runMain scalation.modeling.project2TransformedRegression
//
// Transformed Regression (TranRegression) on 2 of the 3 datasets: Auto MPG and Concrete.
// Model: f(y) = b dot x + e, predictions are mapped back with the inverse f^-1(b dot x),
// so all headline QoF measures below are on the ORIGINAL scale of y (comparable to Regression).

@main def project2TransformedRegression (): Unit =

    Files.createDirectories (Paths.get (trOutDir))
    val datasets = Array (
        ("auto_mpg", "data/auto_mpg.csv"),
        ("concrete", "data/concrete.csv")
    )

    for (name, path) <- datasets do trRunDataset (name, path)

end project2TransformedRegression


private val trLambdas = VectorD.range (0, 13) * 0.25 - 1.0              // Box-Cox grid -1.0, -0.75, ..., 2.0
private val trFolds   = 5
private val trOutDir  = "project2/results/"

// Box-Cox with lambda = 0 is the log transform; BoxcoxForm divides by lambda, so special-case it.
private def trForm (tname: String, lambda: Double = 1.0): Transform = tname match
    case "log"    => LogForm (VectorD (0.0, 1.0))                        // log (y)
    case "sqrt"   => new RootForm (VectorD (0.0, 0.5)):
        override def fi (z: MatrixD): MatrixD = z.map_ (v => if v >= 0 then v * v else Double.NaN)
    case "recip"  => new RecipForm (VectorD (0.0, -1.0)):
        override def fi (z: MatrixD): MatrixD = z.map_ (v => if v > 0 then 1.0 / v else Double.NaN)
    case "boxcox" =>
        if lambda == 0.0 then LogForm (VectorD (0.0, 1.0))
        else new BoxcoxForm (VectorD (lambda)):
            // A nonpositive base is outside the range of Box-Cox for positive y,
            // even when an integer inverse power happens to return a finite value.
            override def fi (z: MatrixD): MatrixD =
                z.map_ (v => if lambda * v + 1 > 0 then math.pow (lambda * v + 1, 1 / lambda) else Double.NaN)
    case _ => throw new IllegalArgumentException (s"Unknown transform: $tname")
end trForm

private def trRows (m: MatrixD, idx: Seq [Int]): MatrixD =
    val sub = new MatrixD (idx.size, m.dim2)
    for (r, i) <- idx.zipWithIndex do sub(i) = m(r)
    sub
end trRows

private def trElems (v: VectorD, idx: Seq [Int]): VectorD =
    val sub = new VectorD (idx.size)
    for (r, i) <- idx.zipWithIndex do sub(i) = v(r)
    sub
end trElems

// Original-scale QoF: (R^2, RMSE, MAE); NaN predictions (invalid inverse) yield NaN measures.
private def trQoF (y: VectorD, yp: VectorD): (Double, Double, Double) =
    if yp.indices.exists (i => !yp(i).isFinite) then return (Double.NaN, Double.NaN, Double.NaN)
    val e   = y - yp
    val sse = e.normSq
    val sst = (y - y.mean).normSq
    (1.0 - sse / sst, math.sqrt (sse / y.dim), e.map (math.abs).sum / y.dim)
end trQoF

private def trFitPredict (xTr: MatrixD, yTr: VectorD, xTe: MatrixD, form: Transform): VectorD =
    val mod = if form == null then new Regression (xTr, yTr)
              else new TranRegression (xTr, yTr, null, Regression.hp, form)
    mod.train ()
    mod.predict (xTe)
end trFitPredict

// k-fold CV on the TRAINING set only; returns the original-scale RMSE for each Box-Cox lambda.
private def trTuneLambda (x: MatrixD, y: VectorD): VectorD =
    val n     = y.dim
    val folds = (0 until n).grouped ((n + trFolds - 1) / trFolds).toArray
    val cvRmse = new VectorD (trLambdas.dim)
    for k <- trLambdas.indices do
        var sse = 0.0
        for fold <- folds do
            val rest = (0 until n).filterNot (fold.contains)
            val yp   = trFitPredict (trRows (x, rest), trElems (y, rest), trRows (x, fold), trForm ("boxcox", trLambdas(k)))
            val e    = trElems (y, fold) - yp
            sse += e.normSq
        end for
        cvRmse(k) = if !sse.isFinite then Double.PositiveInfinity else math.sqrt (sse / n)
    end for
    cvRmse
end trTuneLambda

private def trRunDataset (name: String, path: String): Unit =

    banner (s"Project 2 - Transformed Regression - $name")

    val (xy, colNames) = MatrixD.loadH (path, fullPath = true)
    val p     = xy.dim2 - 1
    val x     = xy(?, 0 until p)
    val y     = xy(?, p)
    val x_    = VectorD.one (x.dim) +^: x
    val fname = "intercept" +: colNames.take (p)
    require (y.indices.forall (i => y(i).isFinite && y(i) > 0), s"$name requires finite positive responses")
    require (x.indices.forall (i => x.indices2.forall (j => x(i, j).isFinite)), s"$name has nonfinite predictors")

    println (s"response = ${colNames(p)}, predictors = ${colNames.take (p).mkString (", ")}")
    println (f"n = ${y.dim}, y in [${y.min}%.3f, ${y.max}%.3f], mean = ${y.mean}%.3f")

    // same 80/20 split scheme as the Ridge/Lasso models (shuffle with seed 42)
    val n        = y.dim
    val nTest    = (n * 0.20).toInt
    val shuffled = new Random (42).shuffle ((0 until n).toList)
    val trainIdx = shuffled.take (n - nTest)
    val testIdx  = shuffled.drop (n - nTest)
    val (xTr, yTr) = (trRows (x_, trainIdx), trElems (y, trainIdx))
    val (xTe, yTe) = (trRows (x_, testIdx), trElems (y, testIdx))
    println (s"Training rows = ${trainIdx.size}; test rows = ${testIdx.size}; shuffle seed = 42")
    val splitOut = new PrintWriter (s"$trOutDir${name}_transformed_split.csv")
    splitOut.println ("row_index,split,cv_fold")
    val foldSize = (trainIdx.size + trFolds - 1) / trFolds
    for (row, i) <- trainIdx.zipWithIndex do splitOut.println (s"$row,train,${i / foldSize}")
    for row <- testIdx do splitOut.println (s"$row,test,-1")
    splitOut.close ()

    banner (s"$name: Box-Cox lambda tuning ($trFolds-fold CV on the training set)")
    val cvRmse  = trTuneLambda (xTr, yTr)
    val best    = cvRmse.argmin ()
    require (cvRmse(best).isFinite, s"No valid Box-Cox candidate for $name")
    val bestLam = trLambdas(best)
    for k <- trLambdas.indices do
        val rmse = if cvRmse(k).isInfinite then "invalid (inverse undefined for some predictions)"
                   else f"${cvRmse(k)}%.4f"
        println (f"lambda = ${trLambdas(k)}%5.2f  CV RMSE = $rmse" + (if k == best then "  <-- best" else ""))
    println (f"\nBest Box-Cox lambda = $bestLam%.2f (lambda = 1 is equivalent to no transformation)")

    val cvOut = new PrintWriter (s"$trOutDir${name}_boxcox_lambda_cv.csv")
    cvOut.println ("lambda,cv_rmse")
    for k <- trLambdas.indices do cvOut.println (s"${trLambdas(k)}," + (if cvRmse(k).isInfinite then "nan" else cvRmse(k)))
    cvOut.close ()

    val models = Array (
        ("Regression (no transform)", null),
        ("TranRegression log",        trForm ("log")),
        ("TranRegression sqrt",       trForm ("sqrt")),
        ("TranRegression reciprocal", trForm ("recip")),
        (f"TranRegression Box-Cox (lambda = $bestLam%.2f)", trForm ("boxcox", bestLam))
    )

    banner (s"$name: model comparison on the ORIGINAL scale of ${colNames(p)}")
    println (f"${"Model"}%-40s ${"In-sample R^2"}%14s ${"RMSE"}%9s ${"MAE"}%9s | ${"Test R^2"}%9s ${"Test RMSE"}%10s ${"Test MAE"}%9s")
    val ypFull = new Array [VectorD] (models.length)
    val ypTest = new Array [VectorD] (models.length)
    val metricsOut = new PrintWriter (s"$trOutDir${name}_transformed_metrics.csv")
    metricsOut.println ("model,split,n,r2,rmse,mae,invalid_predictions")
    for ((mname, form), i) <- models.zipWithIndex do
        ypFull(i) = trFitPredict (x_, y, x_, form)
        ypTest(i) = trFitPredict (xTr, yTr, xTe, form)
        val (r2, rmse, mae)    = trQoF (y, ypFull(i))
        val (r2t, rmset, maet) = trQoF (yTe, ypTest(i))
        println (f"$mname%-40s $r2%14.4f $rmse%9.4f $mae%9.4f | $r2t%9.4f $rmset%10.4f $maet%9.4f")
        metricsOut.println (s"$mname,full_fit,${y.dim},$r2,$rmse,$mae,${ypFull(i).indices.count (j => !ypFull(i)(j).isFinite)}")
        metricsOut.println (s"$mname,test,${yTe.dim},$r2t,$rmset,$maet,${ypTest(i).indices.count (j => !ypTest(i)(j).isFinite)}")
    end for
    metricsOut.close ()

    banner (f"$name: best model fit report - Box-Cox lambda = $bestLam%.2f")
    val mod = new TranRegression (x_, y, fname, Regression.hp, trForm ("boxcox", bestLam))
    mod.train ()
    val (_, qofT) = mod.test_ ()                                          // transformed scale: needed for summary stats
    println ("QoF on the TRANSFORMED scale:")
    println (mod.report (qofT))
    println ("Coefficients are on the TRANSFORMED scale (standard errors use the transformed-scale MSE):")
    println (mod.summary ())
    val (_, qofO) = mod.test ()                                           // original scale: comparable to Regression
    println ("QoF on the ORIGINAL scale:")
    println (mod.report (qofO))

    val predOut = new PrintWriter (s"$trOutDir${name}_transformed_predictions.csv")
    predOut.println ("y,yp_regression,yp_log,yp_sqrt,yp_recip,yp_boxcox")
    for i <- y.indices do predOut.println ((y(i) +: ypFull.map (_(i))).mkString (","))
    predOut.close ()
    val testOut = new PrintWriter (s"$trOutDir${name}_transformed_test_predictions.csv")
    testOut.println ("row_index,y,yp_regression,yp_log,yp_sqrt,yp_recip,yp_boxcox")
    for i <- yTe.indices do testOut.println (s"${testIdx(i)}," + (yTe(i) +: ypTest.map (_(i))).mkString (","))
    testOut.close ()

end trRunDataset

// Check the inverse boundaries that ordinary floating-point power evaluation misses.
@main def project2TransformedDomainCheck (): Unit =
    for lambda <- Seq (-1.0, -0.5, 0.0, 0.5, 1.0, 1.25) do
        val form = trForm ("boxcox", lambda)
        for y <- Seq (0.1, 1.0, 20.0) do
            assert (math.abs (form.fi_(form.f_(y)) - y) < 1E-8)
        if lambda != 0 then
            assert (form.fi_(-1.0 / lambda).isNaN)
            assert (form.fi_(-2.0 / lambda).isNaN)
    assert (trForm ("sqrt").fi_(-1.0).isNaN)
    assert (trForm ("recip").fi_(0.0).isNaN)
    assert (trForm ("recip").fi_(-1.0).isNaN)
    println ("Transformed regression inverse-domain checks passed")
end project2TransformedDomainCheck
