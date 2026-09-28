package scalation
package modeling

import scalation.*
import scalation.mathstat.*
import scala.util.Random

// To execute: runMain scalation.modeling.project2_Forward

// shuffle a matrix's rows in memory using a fixed seed
private def shuffleRows (xy: MatrixD, seed: Int = 42): MatrixD =
  val idx = Random (seed).shuffle (Range (0, xy.dim).toIndexedSeq).toArray
  xy(idx)

private def loadFS (key: String): (MatrixD, VectorD, Array [String]) =
  val (xyRaw, cols) = MatrixD.loadH (s"data/${key}.csv", sp = ',', fullPath = true)
  val xy = shuffleRows (xyRaw)                     // shuffle rows before splitting x/y
  val p  = xy.dim2 - 1
  (xy(?, 0 until p), xy(?, p), cols.take (p))

@main def project2_Forward (): Unit =

  // collect summary rows to print at the very end (one per dataset)
  case class Summary (key: String, bestNames: Array [String], nVars: Int,
                      rSqBar: Double, rmse: Double, r2: Double)
  val summaries = scala.collection.mutable.ArrayBuffer [Summary] ()

  for key <- Array ("auto_mpg", "concrete") do

    // =====================================================================
    // FORWARD SELECTION
    // =====================================================================

    banner (s"FORWARD SELECTION: ${key.toUpperCase}")

    val (x, y, xn) = loadFS (key)
    val ox    = VectorD.one (x.dim) +^: x            // add intercept column
    val oname = Array ("intercept") ++ xn
    val mod   = new Regression (ox, y, oname)

    val (cols, rSq) = mod.forwardSelAll ()(using QoF.rSqBar.ordinal)
    val order = cols.toArray                          // variable added at each step, in order

    // ---- selection table --------------------------------------------------
    println ()
    println (f"${"step"}%-6s${"variable added"}%-18s${"R^2"}%9s${"adjR^2"}%9s${"sMAPE"}%9s${"R^2cv"}%9s")
    println ("-" * 60)
    for (r, k) <- rSq.zipWithIndex do
      val varName = oname (order (k))
      println (f"$k%-6d${varName}%-18s${r(0)}%9.3f${r(1)}%9.3f${r(2)}%9.3f${r(3)}%9.3f")
    end for

    // x-axis = number of predictors actually in the model at each step (includes intercept):
    // step 0 -> 1 predictor (intercept only), step 1 -> 2 predictors, etc.
    val nVars = VectorD (for k <- rSq.indices yield (k + 1).toDouble)

    new PlotM (nVars, rSq, Regression.metrics, s"R^2 vs n: $key", lines = true)

    // =====================================================================
    // BEST MODEL (by adjusted R^2)
    // =====================================================================

    banner (s"BEST MODEL: ${key.toUpperCase}")

    val bestCols  = mod.getBest.mod_cols
    val bestNames = newFname (oname, bestCols)
    println (s"Variables selected (${bestNames.length}): ${bestNames.mkString (", ")}")

    val fin = new Regression (ox(?, bestCols), y, bestNames)

    // trainNtest both trains/reports AND triggers the black/actual vs red/predicted
    // plot -- train() alone (what this script did before) never calls that plot.
    val (yp, qof) = fin.trainNtest ()()

    println ()
    println ("Coefficients:")
    for j <- bestNames.indices do
      println (f"${bestNames (j)}%-15s${fin.parameter (j)}%12.6f")
    end for

    println ()
    println ("Actual vs predicted (first 5):")
    for i <- 0 until math.min (5, y.dim) do
      println (f"Actual: ${y (i)}%.3f  Predicted: ${yp (i)}%.3f")
    end for

    val rmse = qof (QoF.rmse.ordinal)
    val r2   = qof (QoF.rSq.ordinal)

    println ()
    println (f"RMSE:      $rmse%.6f")
    println (f"R^2:       $r2%.6f")
    println (f"adjR^2:    ${rSq (bestCols.size - 1)(1)}%.6f  (from selection step above)")

    summaries += Summary (key, bestNames, bestNames.length,
      rSq (bestCols.size - 1)(1), rmse, r2)
  end for

  // =========================================================================
  // SUMMARY
  // =========================================================================

  banner ("SUMMARY")

  for s <- summaries do
    println (s.key.toUpperCase)
    println (f"  # predictors = ${s.nVars}")
    println (s"  variables    = ${s.bestNames.mkString (", ")}")
    println (f"  adjR^2       = ${s.rSqBar}%.6f")
    println (f"  RMSE         = ${s.rmse}%.6f")
    println (f"  R^2          = ${s.r2}%.6f")
    println ()
  end for

end project2_Forward