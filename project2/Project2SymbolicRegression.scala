package scalation
package modeling 
import scalation.mathstat.*
import scala.collection.mutable.{LinkedHashSet => LSET}

@main def project2Symbolic(): Unit = 
    val data = MatrixD.load("auto_mpg.csv", 1, 0)
    println(s"Rows: ${data.dim}")
    println(s"Columns: ${data.dim2}")
    
    val targetColumn = data.dim2 - 1

    val y = data(?, targetColumn)
    val x = data.not(?, targetColumn)

    println(s"Target column index: $targetColumn")
    println(s"Number of values in y: ${y.dim}")
    println(s"Number of predictor columns in x: ${x.dim2}")

    val featureNames = Array(
        "displacement",
        "cylinders",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin"
        )
    val powers = LSET(-2.0, -1.0, 0.5, 2.0)

    println(s"Transformation powers: ${powers.mkString(", ")}")
    println(s"Predictors: ${featureNames.mkString(", ")}")

    val model = SymbolicRegression(
            x, 
            y, 
            featureNames, 
            powers,
            intercept = true,
            cross = false
        )
    println("Symbolic regression model created.")

    val (_, testMetrics) = model.validate(
        rando = true, 
        ratio = 0.20
    )()
    
    val (selectedColumns, _) = model.selectFeatures(
        SelectionTech.Forward,
        "one"
    )
    println(s"Selected term indexes: ${selectedColumns.mkString(", ")}")

    val selectedModel = model.getBest.mod

    val (_, selectedMetrics) = selectedModel.validate(
        rando = true, 
        ratio = 0.20
    )()
    println(selectedModel.summary())
    println(s"Selected-model test metrics: $selectedMetrics")
    

    // println(model.summary())
    println(s"Test metrics: $testMetrics") 
    //prediction stores the predicted MPG Values for test cars
    //test metrics stores measurements such as R^2 and error stats


end project2Symbolic