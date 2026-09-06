//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @note Project 1: EDA and Simple Regression for three UCI datasets.
 */

package scalation
package modeling

import java.io.File
import scala.math.abs
import scala.runtime.ScalaRunTime.stringOf

import scalation.database.table.Table
import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Metadata for one processed numeric CSV file.
 */
case class ProjectDataset (key: String, title: String, csv: String, target: String, columns: Array [String])


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The three Project 1 datasets (auto_mpg, concrete, airfoil) and their metadata,
 *  shared by every `@main` entry point below.
 */
private val datasets = Array (
    ProjectDataset ("auto_mpg", "Auto MPG", "project1/data/processed/auto_mpg.csv", "mpg",
        Array ("cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "mpg")),
    ProjectDataset ("concrete", "Concrete Compressive Strength", "project1/data/processed/concrete.csv", "compressive_strength_mpa",
        Array ("cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age", "compressive_strength_mpa")),
    ProjectDataset ("airfoil", "Airfoil Self-Noise", "project1/data/processed/airfoil.csv", "scaled_sound_pressure_db",
        Array ("frequency_hz", "angle_attack_deg", "chord_length_m", "free_stream_velocity_ms", "suction_displacement_thickness_m", "scaled_sound_pressure_db"))
)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Load the Project 1 CSV files with `MatrixD.load`, print EDA summaries, and run
 *  `SimpleRegression` for the two predictors with the largest absolute target
 *  correlations. Also demonstrates `Table.load` and documents `MatrixD.loadStr`.
 *
 *  > runMain scalation.modeling.project1EDA
 */
@main def project1EDA (): Unit =

    for ds <- datasets do analyze (ds, makePlots = false)

    demoDataLoadingTechniques ()

end project1EDA


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Display the correlation matrix of each of the three datasets with ScalaTion's
 *  built-in `HeatMap`. `MatrixD.loadH` reads the numeric data together with its
 *  CSV column names.
 *
 *  > runMain scalation.modeling.project1HeatMaps
 */
@main def project1HeatMaps (): Unit =

    for ds <- datasets do
        val filePath = resolvePath (ds)
        val (xy, columnNames) = MatrixD.loadH (filePath, sp = ',', fullPath = true)
        val heatMap = new HeatMap (xy.corr, columnNames, s"${ds.title}: Correlation HeatMap")
        println (s"${ds.title} correlation heat map = $heatMap")
    end for

end project1HeatMaps


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Run the two `SimpleRegression`s for each of the three datasets and open the
 *  ScalaTion plots of observed `y` and fitted `y-hat` versus each selected predictor.
 *
 *  > runMain scalation.modeling.project1RegressionPlots
 */
@main def project1RegressionPlots (): Unit =

    for ds <- datasets do analyze (ds, makePlots = true)

end project1RegressionPlots


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Resolve the file path across common project locations.
 */
private def resolvePath (ds: ProjectDataset): String =
    val candidates = Array (
        ds.csv,
        s"project1/data/processed/${ds.key}.csv",
        s"project1_eda/data/processed/${ds.key}.csv",
        s"project1/${ds.key}.csv",
        s"data/${ds.key}.csv",
        s"${System.getProperty ("user.home")}/Downloads/${ds.key}.csv"
    )
    candidates.find (p => new File (p).exists ()).getOrElse (ds.csv)
end resolvePath


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Analyze one dataset.
 *  @param ds         dataset metadata
 *  @param makePlots  whether to open ScalaTion GUI plots
 */
private def analyze (ds: ProjectDataset, makePlots: Boolean): Unit =

    banner (ds.title)
    val filePath = resolvePath (ds)
    println (s"Loading dataset from: $filePath")
    val xy = MatrixD.load (filePath, skip = 1, sp = ',', fullPath = true)
    val p  = xy.dim2 - 1
    val x  = xy(?, 0 until p)
    val y  = xy(?, p)

    println (s"shape = ${xy.dim}-by-${xy.dim2}, target = ${ds.target}")
    println (s"columns = ${stringOf (ds.columns)}")

    banner (s"${ds.title}: Statistical Summaries")
    for j <- xy.indices2 do
        val v = xy(?, j)
        println (f"${ds.columns(j)}%-38s mean=${v.mean}%10.4f  std=${v.stdev}%10.4f  min=${v.min}%10.4f  max=${v.max}%10.4f")
    end for

    banner (s"${ds.title}: Correlation Matrix")
    println (xy.corr)

    val ranked = (0 until p).map (j => (j, x(?, j).corr (y))).sortBy { case (_, r) => -abs (r) }
    println ("target correlations = " + ranked.map { case (j, r) => f"${ds.columns(j)}=$r%.4f" }.mkString (", "))

    for (j, r) <- ranked.take (2) do
        val feature = ds.columns(j)
        banner (s"${ds.title}: SimpleRegression using $feature")
        val mod = SimpleRegression (x(?, j), y, Array ("one", feature))
        mod.train ()
        val (yp, qof) = mod.test ()
        val b0 = mod.parameter(0)
        val b1 = mod.parameter(1)
        val r2 = mod.fit(QoF.rSq.ordinal)
        val adjR2 = mod.fit(QoF.rSqBar.ordinal)
        val rmse = mod.fit(QoF.rmse.ordinal)

        println (s"target correlation = $r")
        println (s"qof = $qof")
        println (mod.summary ())

        val sign = if b1 >= 0.0 then "+" else "-"
        println (s"--- SUMMARY STATEMENT ---")
        println (f"Dataset: ${ds.title} | Predictor: $feature -> Target: ${ds.target}")
        println (f"Estimated Model: ${ds.target} = $b0%.4f $sign ${abs (b1)}%.4f * $feature")
        println (f"Correlation: r = $r%.4f | R^2 = $r2%.4f | Adj R^2 = $adjR2%.4f | RMSE = $rmse%.4f")
        println (f"Interpretation : Each 1-unit increase in $feature is associated with a ${b1}%.4f change in ${ds.target}.")
        println ("-------------------------\n")

        if makePlots then new Plot (x(?, j), y, yp, s"${ds.title}: y and y-hat vs. $feature", lines = true)
    end for

end analyze


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Demonstrate and explain alternative ScalaTion CSV loading mechanisms:
 *  1. MatrixD.load: purely numeric matrices (default for numerical analysis).
 *  2. MatrixD.loadStr: text files with string columns mapped to ordinal integer codes.
 *  3. Table.load: relational tables with schema metadata and column projection.
 */
private def demoDataLoadingTechniques (): Unit =

    banner ("Demonstration: ScalaTion CSV Loading Techniques")
    println ("""
    * 1. MatrixD.load:
    *    Reads numeric matrices directly from CSV/delimited text into MatrixD.
    *    Example: val xy = MatrixD.load ("data/auto_mpg.csv", skip = 1, sp = ',', fullPath = true)
    *
    * 2. MatrixD.loadStr:
    *    Reads CSV files containing string columns and converts specified string columns
    *    into ordinal integer codes based on provided VectorS dictionaries.
    *    Example: val xy = MatrixD.loadStr ("data.csv", skip = 1, skipCol = 0)(Set (colIdx), dictVectorS)
    *
    * 3. Table.load:
    *    Loads tabular CSV data into a relational Table instance in scalation.database.table,
    *    preserving column names and allowing conversion to MatrixD via data.toMatrixV (...).
    *    Example: val tab = Table.load ("auto_mpg.csv", "auto_mpg", 8, null)
    """.stripMargin)

    val autoLocal = "data/auto_mpg.csv"
    if new File (autoLocal).exists () then
        try
            val tab = Table.load ("auto_mpg.csv", "auto_mpg", 8, null)
            println (s"Table.load successfully loaded '${tab.name}' with ${tab.rows} rows and schema: ${tab.schema.mkString (", ")}")
        catch
            case ex: Throwable => println (s"Note on Table.load demo: ${ex.getMessage}")
    end if

end demoDataLoadingTechniques
