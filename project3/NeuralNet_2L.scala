package scalation
package modeling
package neuralnet

import scalation.mathstat._
import scalation.modeling.ActivationFun._
import scalation.modeling.neuralnet.Optimizer
import scala.io.Source
import scala.util.Random


object NeuralNet2LProject:

    // store predictors and targets
    case class Dataset(
        x: Array[Array[Double]],
        y: Array[Double]
    )

    // store model evaluation metrics
    case class QoF(r2: Double, mse: Double, rmse: Double)

    // store selected hyperparameters
    case class BestParams(
        learningRate: Double,
        batchSize: Int,
        validationMSE: Double
    )

    // store QoF results for all datasets
    val qofResults = scala.collection.mutable.ArrayBuffer[(String, QoF, QoF)]()


    // load numerical csv data
    def loadDataset(path: String, target: String): Dataset =
        val source = Source.fromFile(path)

        try
            val lines = source.getLines().toArray
            val headers = lines.head.split(",").map(_.trim)
            val targetIndex = headers.indexOf(target)

            require(targetIndex >= 0, s"Target not found: $target")

            val rows = lines.drop(1).filter(_.trim.nonEmpty).map { line =>
                line.split(",", -1).map(_.trim)
            }

            val data = rows.map { row =>
                require(
                    row.length == headers.length,
                    "Unexpected number of CSV columns"
                )
                row.map(_.toDouble)
            }

            val x = data.map { row =>
                row.indices
                    .filter(_ != targetIndex)
                    .map(row(_))
                    .toArray
            }

            val y = data.map(_(targetIndex))

            Dataset(x, y)
        finally
            source.close()
    end loadDataset


    // split observations into training and validation sets
    def splitData(
        data: Dataset,
        validationRatio: Double,
        seed: Int = 42
    ): (Dataset, Dataset) =

        val indices = new Random(seed).shuffle(data.y.indices.toList)
        val validationSize = math.ceil(data.y.length * validationRatio).toInt
        val valIndices = indices.take(validationSize)
        val trainIndices = indices.drop(validationSize)

        def select(rows: List[Int]): Dataset =
            Dataset(
                rows.map(data.x(_)).toArray,
                rows.map(data.y(_)).toArray
            )

        (select(trainIndices), select(valIndices))
    end splitData


    // calculate training means and standard deviations
    def fitScaler(
        x: Array[Array[Double]]
    ): (Array[Double], Array[Double]) =

        val n = x(0).length
        val m = x.length
        val means = Array.ofDim[Double](n)
        val stds = Array.ofDim[Double](n)

        for j <- 0 until n do
            means(j) = x.map(_(j)).sum / m

            val variance = x.map { row =>
                math.pow(row(j) - means(j), 2)
            }.sum / m

            stds(j) = math.sqrt(variance)

            if stds(j) == 0.0 then
                stds(j) = 1.0
        end for

        (means, stds)
    end fitScaler


    // standardize predictors and add intercept column
    def prepareX(
        x: Array[Array[Double]],
        means: Array[Double],
        stds: Array[Double]
    ): MatrixD =

        val matrix = new MatrixD(
            x.length,
            x(0).length + 1,
            Array.ofDim[Double](x.length, x(0).length + 1)
        )

        for i <- x.indices do
            matrix(i, 0) = 1.0

            for j <- x(i).indices do
                matrix(i, j + 1) =
                    (x(i)(j) - means(j)) / stds(j)
            end for
        end for

        matrix
    end prepareX


    // convert target values into a scalation matrix
    def prepareY(y: Array[Double]): MatrixD =
        val matrix = new MatrixD(y.length, 1)

        for i <- y.indices do
            matrix(i, 0) = y(i)
        end for

        matrix
    end prepareY


    // calculate r², mse, and rmse
    def calculateQoF(
        actual: MatrixD,
        predicted: MatrixD
    ): QoF =

        val m = actual.dim

        val mean = (0 until m).map { i =>
            actual(i, 0)
        }.sum / m

        val sse = (0 until m).map { i =>
            math.pow(actual(i, 0) - predicted(i, 0), 2)
        }.sum

        val sst = (0 until m).map { i =>
            math.pow(actual(i, 0) - mean, 2)
        }.sum

        val mse = sse / m
        val rmse = math.sqrt(mse)

        val r2 =
            if sst > 0.0 then 1.0 - sse / sst
            else Double.NaN

        QoF(r2, mse, rmse)
    end calculateQoF


    // create and train a 2L neural network
    def trainModel(
        x: MatrixD,
        y: MatrixD,
        lr: Double,
        batchSize: Int
    ): NeuralNet_2L =

        Optimizer.hp("eta") = lr
        Optimizer.hp("bSize") = batchSize.toDouble

        // identity activation for regression
        val model = new NeuralNet_2L(x, y, f = f_id)

        // train using scalation's built-in optimizer
        model.train()

        model
    end trainModel


    // tune hyperparamters (learning rate and mini-batch size)
    def hyperparameterTuning(data: Dataset): BestParams =

        // validation split
        val (train, validation) = splitData(data, 0.20)

        // standardize predictors using tuning training data
        val (means, stds) = fitScaler(train.x)
        val xTrain = prepareX(train.x, means, stds)
        val yTrain = prepareY(train.y)
        val xVal = prepareX(validation.x, means, stds)
        val yVal = prepareY(validation.y)

        val learningRates = Array(0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3)
        val batchSizes = Array(16, 32, 64)
        var bestMSE = Double.PositiveInfinity
        var bestLR = 0.0
        var bestBatch = 0

        // test all hyperparameter combinations
        for lr <- learningRates do
            for batch <- batchSizes do

                println(s"Testing LR=$lr, Batch=$batch")

                val model = trainModel(
                    xTrain, yTrain, lr, batch
                )

                // calculate validation mse
                val predictions = model.predict(xVal)
                val qof = calculateQoF(yVal, predictions)

                println(f"Validation MSE = ${qof.mse}%.4f")

                if !qof.mse.isNaN && qof.mse < bestMSE then
                    bestMSE = qof.mse
                    bestLR = lr
                    bestBatch = batch
                end if

            end for
        end for

        val best = BestParams(bestLR, bestBatch, bestMSE)
        println(s"Best hyperparameters: $best")
        best
    end hyperparameterTuning


    // plot actual vs. predicted values
    def plotPredictions(
        actual: MatrixD,
        predicted: MatrixD,
        title: String
    ): Unit =

        val y = actual(?, 0)
        val yp = predicted(?, 0)

        new Plot(y, yp, null, title, lines = false)
    end plotPredictions


    // plot residuals against predicted values
    def plotResiduals(
        actual: MatrixD,
        predicted: MatrixD,
        title: String
    ): Unit =

        val y = actual(?, 0)
        val yp = predicted(?, 0)
        val residuals = y - yp

        new Plot(yp, residuals, null, title, lines = false)
    end plotResiduals


    // run the complete experiment for one dataset
    def runExperiment(
        name: String,
        path: String,
        target: String
    ): BestParams =

        println(s"\n========== $name ==========")

        // load dataset
        val data = loadDataset(path, target)
        println(s"Observations: ${data.y.length}")
        println(s"Predictors: ${data.x(0).length}")

        // reserve 20% for final evaluation
        val (train, validation) = splitData(data, 0.20)

        // tune using training data only
        val best = hyperparameterTuning(train)

        // standardize final training and validation predictors
        val (means, stds) = fitScaler(train.x)
        val xTrain = prepareX(train.x, means, stds)
        val yTrain = prepareY(train.y)
        val xVal = prepareX(validation.x, means, stds)
        val yVal = prepareY(validation.y)

        // train final model using selected hyperparameters
        val model = trainModel(
            xTrain,
            yTrain,
            best.learningRate,
            best.batchSize
        )

        // generate predictions
        val trainPred = model.predict(xTrain)
        val valPred = model.predict(xVal)

        // calculate in-sample and validation QoF
        val trainQoF = calculateQoF(yTrain, trainPred)
        val valQoF = calculateQoF(yVal, valPred)
        
        // save training and validation QoF
        qofResults += ((name, trainQoF, valQoF))

        println(s"\n$name - Training QoF")
        println(s"R² = ${trainQoF.r2}")
        println(s"MSE = ${trainQoF.mse}")
        println(s"RMSE = ${trainQoF.rmse}")

        println(s"\n$name - Validation QoF")
        println(s"R² = ${valQoF.r2}")
        println(s"MSE = ${valQoF.mse}")
        println(s"RMSE = ${valQoF.rmse}")

        // plot training loss against epochs
        model.opti.plotLoss(s"$name - 2L Loss")

        // actual vs. predicted plots
        plotPredictions(
            yTrain, trainPred, s"$name - Training Predictions"
        )

        plotPredictions(
            yVal, valPred, s"$name - Validation Predictions"
        )

        // residual plots
        plotResiduals(
            yTrain, trainPred, s"$name - Training Residuals"
        )

        plotResiduals(
            yVal, valPred, s"$name - Validation Residuals"
        )
        
        best
    end runExperiment


    // run all three datasets
    @main def runNeuralNet2L(): Unit =

        val airfoil = runExperiment(
            "Airfoil",
            "data/airfoil.csv",
            "scaled_sound_pressure_db"
        )

        val auto = runExperiment(
            "Auto MPG",
            "data/auto_mpg.csv",
            "mpg"
        )

        val concrete = runExperiment(
            "Concrete",
            "data/concrete.csv",
            "compressive_strength_mpa"
        )

        // display best hyperparameters for all datasets
        println("\n========== HYPERPARAMETER TUNING RESULTS ==========")
        println("Dataset     Learning Rate    Batch Size    Validation MSE")

        for (name, best) <- Seq(
            ("Airfoil", airfoil),
            ("Auto MPG", auto),
            ("Concrete", concrete)
        ) do
            println(f"$name%-12s ${best.learningRate}%-16.2f ${best.batchSize}%-13d ${best.validationMSE}%.4f")
        end for

        // display QoF comparison table
        println("\n========== 2L NEURAL NETWORK - QUALITY OF FIT ==========")
        println(f"${"Dataset"}%-12s ${"Train R²"}%-12s ${"Validation R²"}%-16s ${"Train MSE"}%-12s ${"Validation MSE"}%-17s ${"Train RMSE"}%-13s ${"Validation RMSE"}")

        for (name, train, validation) <- qofResults do
            println(
                f"$name%-12s ${train.r2}%-12.4f ${validation.r2}%-16.4f " +
                f"${train.mse}%-12.4f ${validation.mse}%-17.4f " +
                f"${train.rmse}%-13.4f ${validation.rmse}%.4f"
            )
        end for

    end runNeuralNet2L

end NeuralNet2LProject
