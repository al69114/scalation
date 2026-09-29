// Diagnostic reconstruction, not the missing original experiment runner.
// Uses existing ScalaTion classes without changing library defaults on disk.
package scalation.modeling

import scalation.*
import scalation.mathstat.*
import scala.collection.mutable.{LinkedHashSet => LSET}
import java.nio.file.{Files, Paths}

@main def checkSymbolicData (): Unit =
  FeatureSelection.fullset_FS = false
  val output = Paths.get("project2/results/symbolic_audit")
  Files.createDirectories(output)
  for key <- Array("auto_mpg", "concrete", "airfoil") do
    val (xy, names) = MatrixD.loadH(s"data/$key.csv", sp = ',', fullPath = true)
    val y = xy(?, xy.dim2 - 1)
    val pg = TnT_Split.makePermGen(y.dim)
    val first = TnT_Split.testIndices(pg, (y.dim * 0.2).toInt)
    val second = TnT_Split.testIndices(pg, (y.dim * 0.2).toInt)
    Files.writeString(output.resolve(s"${key}_second_split.csv"),
      "row_index,y\n" + second.map(i => s"$i,${y(i)}").mkString("\n") + "\n")
    println(s"AUDIT $key first-test variance ${(y(first) - y(first).mean).normSq / first.size}")
    println(s"AUDIT $key second-test variance ${(y(second) - y(second).mean).normSq / second.size}")
    if key != "airfoil" then
      val nc = xy.dim2 - 1
      val x = xy(?, 0 until nc)
      val powers = if key == "auto_mpg" then LSET(-2.0,-1.0,0.5,2.0)
                   else LSET(1.0,0.5,2.0)
      val mod = SymbolicRegression(x, y, names.take(nc), powers, cross = false)
      val (_, candidateQof) = mod.validate()()
      println(s"AUDIT $key candidate ${FitM.fitMap(candidateQof, QoF.values.map(_.toString))}")
      // The default criterion is sMAPE_IC, not adjusted R-squared.
      mod.forwardSelAll("many")(using QoF.smapeC.ordinal)
      val best = mod.getBest.mod.asInstanceOf[Regression]
      // CV consumed one permutation. Default validate now uses the second.
      val (yp, qof) = best.validate()()
      println(s"AUDIT $key selected ${FitM.fitMap(qof, QoF.values.map(_.toString))}")
      Files.writeString(output.resolve(s"${key}_selected_coefficients.csv"),
        "term,coefficient\n" + best.getFname.indices.map(j =>
          s"${best.getFname(j)},${best.parameter(j)}").mkString("\n") + "\n")
      // TnT_Split preserves original row order when splitting the matrix.
      val ordered = second.sorted
      Files.writeString(output.resolve(s"${key}_selected_predictions.csv"),
        "row_index,y,prediction\n" + ordered.indices.map(j =>
          s"${ordered(j)},${y(ordered(j))},${yp(j)}").mkString("\n") + "\n")
      for j <- best.getFname.indices do
        println(f"AUDITCOEF $key ${best.getFname(j)} ${best.parameter(j)}%.10f")
