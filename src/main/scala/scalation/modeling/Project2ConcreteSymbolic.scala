package scalation
package modeling

import scalation.mathstat.*
import scala.collection.mutable.{LinkedHashSet => LSET}

@main def project2ConcreteSymbolic(): Unit =

    val data = MatrixD.load("concrete.csv", 1, 0)

    val targetColumn = data.dim2 - 1

    val y = data(?, targetColumn)
    val x = data.not(?, targetColumn)

    println(s"Rows: ${data.dim}")
    println(s"Columns: ${data.dim2}")
    println(s"Target column index: $targetColumn")
    println(s"Number of target values: ${y.dim}")
    println(s"Number of predictors: ${x.dim2}")

    val featureNames = Array(
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age"
    )

    val powers = LSET(0.5, 1.0, 2.0)

    println(s"Predictors: ${featureNames.mkString(", ")}")
    println(s"Transformation powers: ${powers.mkString(", ")}")

    val model = SymbolicRegression(
    x,
    y,
    featureNames,
    powers,
    intercept = true,
    cross = false
    )

    println("Concrete symbolic regression model created.")

    val (_, testMetrics) = model.validate(
    rando = true,
    ratio = 0.20
    )()
    val (selectedColumns, _) =
    model.selectFeatures(SelectionTech.Forward, "one")

    println(s"Selection path: ${selectedColumns.mkString(", ")}")

    val selectedModel = model.getBest.mod

    val (_, selectedMetrics) = selectedModel.validate(
    rando = true,
    ratio = 0.20
    )()

    println(selectedModel.summary())
    println(s"Selected-model test metrics: $selectedMetrics") 
    println(s"Test metrics: $testMetrics")
end project2ConcreteSymbolic