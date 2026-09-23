package scalation
package modeling

import scala.util.Random
import scalation.mathstat._
import java.io.{ByteArrayOutputStream, PrintStream}

// To execute: runMain scalation.modeling.project2RegularizedRegression

@main def project2RegularizedRegression (): Unit =

    // -------------------------------------------------------------------------
    // SETUP
    // -------------------------------------------------------------------------

    // load cleaned data from project 1
    val filePath = "data/auto_mpg.csv"

    val xy = MatrixD.load(
        filePath,
        skip = 1,
        sp = ',',
        fullPath = true
    )

    // separate predictors and target
    val x = xy(?, 0 until xy.dim2 - 1)
    val y = xy(?, xy.dim2 - 1)

    val featureNames = Array(
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin"
    )

    // split data into 80% training and 20% testing
    val n = x.dim
    val testSize = (n * 0.20).toInt
    val trainSize = n - testSize

    // shuffle rows before splitting
    val random = new Random(42)
    val indices = random.shuffle((0 until n).toList)

    val trainIdx = indices.take(trainSize)
    val testIdx = indices.drop(trainSize)

    val xTrain = new MatrixD(trainSize, x.dim2)
    val yTrain = new VectorD(trainSize)

    val xTest = new MatrixD(testSize, x.dim2)
    val yTest = new VectorD(testSize)

    // create training set
    for i <- trainIdx.indices do
        val row = trainIdx(i)

        for j <- x.indices2 do
            xTrain(i, j) = x(row, j)
        end for

        yTrain(i) = y(row)
    end for

    // create testing set
    for i <- testIdx.indices do
        val row = testIdx(i)

        for j <- x.indices2 do
            xTest(i, j) = x(row, j)
        end for

        yTest(i) = y(row)
    end for


    // -------------------------------------------------------------------------
    // STANDARDIZE PREDICTORS
    // -------------------------------------------------------------------------

    // calculate training means
    val means = xTrain.mean 

    // calculate training standard deviations
    val stds = new VectorD(xTrain.dim2)

    for j <- xTrain.indices2 do
        stds(j) = xTrain(?, j).stdev
    end for

    val xTrainScaled = xTrain.copy
    val xTestScaled = xTest.copy

    // standardize training data
    for i <- xTrain.indices do
        for j <- xTrain.indices2 do
            xTrainScaled(i, j) =
                (xTrain(i, j) - means(j)) / stds(j)
        end for
    end for

    // standardize test data using training mean/std
    for i <- xTest.indices do
        for j <- xTest.indices2 do
            xTestScaled(i, j) =
                (xTest(i, j) - means(j)) / stds(j)
        end for
    end for


    // =========================================================================
    // RIDGE REGRESSION
    // =========================================================================

    banner("RIDGE REGRESSION")

    // center target using training mean
    val yMean = yTrain.mean
    val yTrainCentered = yTrain - yMean

    // create ridge model for lambda tuning
    val ridgeBase = new RidgeRegression(
        xTrainScaled,
        yTrainCentered,
        featureNames
    )

    // tune ridge lambda using cross-validation
    val ridgeOutput = new ByteArrayOutputStream()

    val (bestRidgeLambda, bestRidgeCVSSE) =
        Console.withOut(new PrintStream(ridgeOutput)) {
            ridgeBase.findLambda
        }

    // display lambda and CV SSE for each candidate
    println("Ridge lambda tuning:")

    ridgeOutput
        .toString
        .linesIterator
        .filter(line =>
            line.contains("RidgeRegression with lambda") &&
            line.contains("sse")
        )
        .foreach(println)

    println(s"\nBest Ridge Lambda: $bestRidgeLambda")
    println(f"Best Ridge CV SSE: $bestRidgeCVSSE%.4f")

    // set best ridge lambda
    val ridgeHP = RidgeRegression.hp.updateReturn(
        "lambda",
        bestRidgeLambda
    )

    // train final ridge model
    val ridgeModel = new RidgeRegression(
        xTrainScaled,
        yTrainCentered,
        featureNames,
        ridgeHP
    )

    ridgeModel.train()

    // display ridge coefficients
    println("\nRidge coefficients:")
    println(f"Intercept = $yMean%.6f")

    for j <- featureNames.indices do
        println(
            f"${featureNames(j)}%-15s ${ridgeModel.parameter(j)}%.6f"
        )
    end for

    // predict MPG for test data
    val ridgePredCentered = ridgeModel.predict(xTestScaled)

    // add target mean back to predictions
    val ridgePred = ridgePredCentered + yMean

    // display actual vs predicted MPG
    println("\nRidge actual vs predicted MPG:")

    for i <- 0 until math.min(5, yTest.dim) do
        println(
            f"Actual: ${yTest(i)}%.3f  Predicted: ${ridgePred(i)}%.3f"
        )
    end for

    // calculate ridge prediction errors
    val ridgeError = yTest - ridgePred

    // calculate RMSE
    val ridgeMSE = ridgeError.normSq / yTest.dim.toDouble
    val ridgeRMSE = math.sqrt(ridgeMSE)

    // calculate R^2
    val ridgeSST = (yTest - yTest.mean).normSq
    val ridgeSSE = ridgeError.normSq
    val ridgeR2 = 1.0 - ridgeSSE / ridgeSST

    println(f"\nRidge RMSE: $ridgeRMSE%.6f")
    println(f"Ridge R²:   $ridgeR2%.6f")


    // =========================================================================
    // LASSO REGRESSION
    // =========================================================================

    banner("LASSO REGRESSION")

    // add intercept column
    val onesTrain = VectorD.one(xTrainScaled.dim)
    val onesTest = VectorD.one(xTestScaled.dim)

    val xTrainLasso = onesTrain +^: xTrainScaled
    val xTestLasso = onesTest +^: xTestScaled

    val lassoFeatureNames =
        Array("const") ++ featureNames

    // create lasso model for lambda tuning
    val lassoBase = new LassoRegression(
        xTrainLasso,
        yTrain,
        lassoFeatureNames
    )

    // tune lasso lambda using cross-validation
    val lassoOutput = new ByteArrayOutputStream()

    val (bestLassoLambda, bestLassoCVSSE) =
        Console.withOut(new PrintStream(lassoOutput)) {
            lassoBase.findLambda
        }

    // display lambda and CV SSE for each candidate
    println("Lasso lambda tuning:")

    lassoOutput
        .toString
        .linesIterator
        .filter(line =>
            line.contains("LassoRegression with lambda") &&
            line.contains("sse")
        )
        .foreach(println)

    println(s"\nBest Lasso Lambda: $bestLassoLambda")
    println(f"Best Lasso CV SSE: $bestLassoCVSSE%.4f")

    // set best lasso lambda
    val lassoHP = LassoRegression.hp.updateReturn(
        "lambda",
        bestLassoLambda
    )

    // train final lasso model
    val lassoModel = new LassoRegression(
        xTrainLasso,
        yTrain,
        lassoFeatureNames,
        lassoHP
    )

    lassoModel.train()

    // display lasso coefficients
    println("\nLasso coefficients:")

    for j <- lassoFeatureNames.indices do
        println(
            f"${lassoFeatureNames(j)}%-15s ${lassoModel.parameter(j)}%.6f"
        )
    end for

    // predict MPG for test data
    val lassoPred = lassoModel.predict(xTestLasso)

    // display actual vs predicted MPG
    println("\nLasso actual vs predicted MPG:")

    for i <- 0 until math.min(5, yTest.dim) do
        println(
            f"Actual: ${yTest(i)}%.3f  Predicted: ${lassoPred(i)}%.3f"
        )
    end for

    // calculate lasso prediction errors
    val lassoError = yTest - lassoPred

    // calculate RMSE
    val lassoMSE = lassoError.normSq / yTest.dim.toDouble
    val lassoRMSE = math.sqrt(lassoMSE)

    // calculate R^2
    val lassoSST = (yTest - yTest.mean).normSq
    val lassoSSE = lassoError.normSq
    val lassoR2 = 1.0 - lassoSSE / lassoSST

    println(f"\nLasso RMSE: $lassoRMSE%.6f")
    println(f"Lasso R²:   $lassoR2%.6f")


    // =========================================================================
    // SUMMARY
    // =========================================================================

    banner("SUMMARY")

    println(f"Ridge lambda = $bestRidgeLambda%.6f")
    println(f"Ridge CV SSE = $bestRidgeCVSSE%.6f")
    println(f"Ridge RMSE   = $ridgeRMSE%.6f")
    println(f"Ridge R²     = $ridgeR2%.6f")

    println()

    println(f"Lasso lambda = $bestLassoLambda%.6f")
    println(f"Lasso CV SSE = $bestLassoCVSSE%.6f")
    println(f"Lasso RMSE   = $lassoRMSE%.6f")
    println(f"Lasso R²     = $lassoR2%.6f")

end project2RegularizedRegression