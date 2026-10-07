package scalation
package modeling

import java.io.PrintWriter
import java.nio.file.{Files, Paths}
import scala.util.Random
import scalation.mathstat._

// Fold preprocessing is fitted inside each CV fold. The intercept is recovered
// from the training response mean and is unpenalized for both Ridge and Lasso.
@main def project2RegularizedRegression (): Unit =
    val out = "project2/results/regularized/"
    Files.createDirectories (Paths.get (out))
    val (xy, header) = MatrixD.loadH ("data/auto_mpg.csv", fullPath = true)
    val p = xy.dim2 - 1
    val x = xy(?, 0 until p)
    val y = xy(?, p)
    val nTest = (y.dim * 0.2).toInt
    val order = new Random (42).shuffle ((0 until y.dim).toList)
    val train = order.take (y.dim - nTest).toIndexedSeq
    val test = order.drop (y.dim - nTest).toIndexedSeq
    val xTr = x(train)
    val yTr = y(train)
    val xTe = x(test)
    val yTe = y(test)
    val folds = (0 until train.size).grouped ((train.size + 4) / 5).map (_.toIndexedSeq).toArray
    val lambdas = Array.tabulate (20)(i => 0.1 * math.pow (2, i))
    def scaled (a: MatrixD, b: MatrixD): (MatrixD, MatrixD) =
        val mu = a.mean
        val sd = VectorD (a.indices2.map (j => a(?, j).stdev).toIndexedSeq)
        require (sd.indices.forall (j => sd(j) > 0))
        val sa = a.copy
        val sb = b.copy
        for j <- a.indices2 do
            sa(?, j) = (a(?, j) - mu(j)) / sd(j)
            sb(?, j) = (b(?, j) - mu(j)) / sd(j)
        (sa, sb)
    def model (method: String, a: MatrixD, b: VectorD, lambda: Double): Predictor =
        if method == "ridge" then new RidgeRegression (a, b, header.take (p),
            RidgeRegression.hp.updateReturn ("lambda", lambda))
        else new LassoRegression (a, b, header.take (p),
            LassoRegression.hp.updateReturn ("lambda", lambda))
    val (sxTr, sxTe) = scaled (xTr, xTe)
    val metricOut = new PrintWriter (s"${out}auto_mpg_metrics.csv")
    metricOut.println ("method,lambda,cv_rmse,test_rmse,test_r2,test_mae")
    val coefOut = new PrintWriter (s"${out}auto_mpg_coefficients.csv")
    coefOut.println ("method,term,coefficient")
    val cvOut = new PrintWriter (s"${out}auto_mpg_tuning.csv")
    cvOut.println ("method,lambda,cv_rmse")
    val predOut = new PrintWriter (s"${out}auto_mpg_test_predictions.csv")
    predOut.println ("method,row_index,y,prediction")
    for method <- Array ("ridge", "lasso") do
        val cv = lambdas.map { lambda =>
            var sse = 0.0
            for validation <- folds do
                val training = (0 until train.size).filterNot (validation.contains).toIndexedSeq
                val (sx, sv) = scaled (xTr(training), xTr(validation))
                val yt = yTr(training)
                val m = model (method, sx, yt - yt.mean, lambda)
                m.train ()
                val prediction = m.predict (sv) + yt.mean
                sse += (yTr(validation) - prediction).normSq
            val rmse = math.sqrt (sse / train.size)
            cvOut.println (s"$method,$lambda,$rmse")
            rmse
        }
        val best = cv.indices.minBy (cv(_))
        val m = model (method, sxTr, yTr - yTr.mean, lambdas(best))
        m.train ()
        val yp = m.predict (sxTe) + yTr.mean
        require (yp.indices.forall (i => yp(i).isFinite))
        val e = yTe - yp
        val rmse = math.sqrt (e.normSq / yTe.dim)
        val r2 = 1 - e.normSq / (yTe - yTe.mean).normSq
        val mae = e.map (math.abs).mean
        metricOut.println (s"$method,${lambdas(best)},${cv(best)},$rmse,$r2,$mae")
        coefOut.println (s"$method,intercept,${yTr.mean}")
        for j <- x.indices2 do coefOut.println (s"$method,${header(j)},${m.parameter(j)}")
        for i <- test.indices do predOut.println (s"$method,${test(i)},${yTe(i)},${yp(i)}")
        println (s"$method: lambda=${lambdas(best)}, CV RMSE=${cv(best)}, test RMSE=$rmse, R2=$r2, MAE=$mae")
    metricOut.close ()
    coefOut.close ()
    cvOut.close ()
    predOut.close ()
    val split = new PrintWriter (s"${out}auto_mpg_split.csv")
    split.println ("row_index,split")
    for i <- train do split.println (s"$i,train")
    for i <- test do split.println (s"$i,test")
    split.close ()
end project2RegularizedRegression
