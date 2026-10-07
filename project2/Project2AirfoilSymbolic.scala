package scalation
package modeling

import scalation.mathstat.*
import scala.collection.mutable.{LinkedHashSet => LSET}

@main def project2AirfoilSymbolic(): Unit =

    // Load the CSV and skip the header
    val data = MatrixD.load("airfoil.csv", 1, 0)

    // The last column is scaled sound pressure level
    val targetColumn = data.dim2 - 1

    // y is the target; x contains the five predictors
    val y = data(?, targetColumn)
    val x = data.not(?, targetColumn)

    println(s"Rows: ${data.dim}")
    println(s"Columns: ${data.dim2}")
    println(s"Target column index: $targetColumn")
    println(s"Number of target values: ${y.dim}")
    println(s"Number of predictors: ${x.dim2}")

    // Names must follow the same order as the CSV columns
    val featureNames = Array(
        "frequency",
        "attack_angle",
        "chord_length",
        "velocity",
        "displacement_thickness"
    )

    // sqrt(x), original x, and x squared
    // Negative powers are avoided because attack_angle may equal zero
    val powers = LSET(0.0, 0.5, 1.0, 2.0)

    println(s"Predictors: ${featureNames.mkString(", ")}")
    println(s"Transformation powers: ${powers.mkString(", ")}")
    
    /* 
    val customTerms: Array[Array[(Int, Double)]] = Array(

    // frequency * displacement_thickness / velocity
    Array((0, 1.0), (4, 1.0), (3, -1.0)),

    // frequency * chord_length / velocity
    Array((0, 1.0), (2, 1.0), (3, -1.0)),

    // chord_length / displacement_thickness
    Array((2, 1.0), (4, -1.0)),

    // PySR-informed candidate:
    // chord_length / (frequency * displacement_thickness^1.5)
    Array((2, 1.0), (0, -1.0), (4, -1.5))
)   */
    // Create the symbolic regression model
    val model = SymbolicRegression(
        x,
        y,
        featureNames,
        powers,
        intercept = true,
        cross = true,
        cross3 = true,
        // chord_length /
    // (frequency * displacement_thickness^1.5)
        terms = Array(
        (2, 1.0),
        (0, -1.0),
        (4, -1.5)
        )
    )
    

    println("Airfoil symbolic regression model created.")

    // Evaluate the complete 15-term model
    val (_, baseMetrics) = model.validate(
        rando = true,
        ratio = 0.20
    )()

    println(s"Base-model test metrics: $baseMetrics")

    // Perform forward selection
    val (selectionPath, _) =
        model.selectFeatures(SelectionTech.Forward, "one")

    println(
        s"Forward-selection path: ${selectionPath.mkString(", ")}"
    )

    // Retrieve the best model found during forward selection
    val selectedModel = model.getBest.mod

    // Evaluate the selected model
    val (_, selectedMetrics) = selectedModel.validate(
        rando = true,
        ratio = 0.20
    )()

    println(selectedModel.summary())
    println(s"Selected-model test metrics: $selectedMetrics")

end project2AirfoilSymbolic