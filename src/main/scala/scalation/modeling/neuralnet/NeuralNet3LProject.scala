package scalation
package modeling
package neuralnet

import scalation.mathstat._
import scalation.modeling.ActivationFun._
import scalation.modeling.neuralnet.Optimizer
import scala.io.Source
import scala.util.Random


object NeuralNet3L:

    case class Dataset(
        x: Array[Array[Double]],
        y: Array[Double]
    )

    case class QoF(r2: Double, mse: Double, rmse: Double)

    case class BestParams(
        hiddenNodes: Int,
        activationName: String,
        learningRate: Double,
        batchSize: Int,
        validationMSE: Double
    )

    val qofResults = scala.collection.mutable.ArrayBuffer[(String, QoF, QoF)]()

    def loadDataset(path: String, target: String): Dataset =
        val source = Source.fromFile(path)
        try
            val lines = source.getLines().toArray
            require(lines.nonEmpty, s"CSV file is empty: $path")
            val headers = lines.head.split(",").map(_.trim)
            val targetIndex = headers.indexOf(target)
            require(targetIndex >= 0, s"Target '$target' not found in $path")

            val rows = lines.drop(1).filter(_.trim.nonEmpty).map { line =>
                line.split(",", -1).map(_.trim)
            }

            val data = rows.map { row =>
                require(row.length == headers.length, s"Unexpected column count in $path")
                row.map(_.toDouble)
            }

            val x = data.map { row =>
                row.indices.filter(_ != targetIndex).map(row(_)).toArray
            }
            val y = data.map(_(targetIndex))
            Dataset(x, y)
        finally
            source.close()
    end loadDataset

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
            Dataset(rows.map(data.x(_)).toArray, rows.map(data.y(_)).toArray)

        (select(trainIndices), select(valIndices))
    end splitData

    def fitScaler(x: Array[Array[Double]]): (Array[Double], Array[Double]) =
        val n = x(0).length
        val m = x.length
        val means = Array.ofDim[Double](n)
        val stds = Array.ofDim[Double](n)

        for j <- 0 until n do
            means(j) = x.map(_(j)).sum / m
            val variance = x.map(row => math.pow(row(j) - means(j), 2)).sum / m
            stds(j) = math.sqrt(variance)
            if stds(j) == 0.0 then stds(j) = 1.0
        end for
        (means, stds)
    end fitScaler

    def prepareX(
        x: Array[Array[Double]],
        means: Array[Double],
        stds: Array[Double]
    ): MatrixD =
        val matrix = new MatrixD(
            x.length,
            x(0).length,
            Array.ofDim[Double](x.length, x(0).length)
        )
        for i <- x.indices do
            for j <- x(i).indices do
                matrix(i, j) = (x(i)(j) - means(j)) / stds(j)
            end for
        end for
        matrix
    end prepareX

    def prepareY(y: Array[Double]): MatrixD =
        val matrix = new MatrixD(y.length, 1)
        for i <- y.indices do matrix(i, 0) = y(i)
        matrix
    end prepareY

    def calculateQoF(actual: MatrixD, predicted: MatrixD): QoF =
        val m = actual.dim
        val mean = (0 until m).map(i => actual(i, 0)).sum / m
        val sse = (0 until m).map(i => math.pow(actual(i, 0) - predicted(i, 0), 2)).sum
        val sst = (0 until m).map(i => math.pow(actual(i, 0) - mean, 2)).sum
        val mse = sse / m
        val r2 = if sst > 0.0 then 1.0 - sse / sst else Double.NaN
        QoF(r2, mse, math.sqrt(mse))
    end calculateQoF

    def trainModel(
        x: MatrixD,
        y: MatrixD,
        hiddenNodes: Int,
        activation: AFF,
        lr: Double,
        batchSize: Int
    ): NeuralNet_3L =
        Optimizer.hp("eta") = lr
        Optimizer.hp("bSize") = batchSize.toDouble
        val model = new NeuralNet_3L(
            x, y,
            nz = hiddenNodes,
            f = activation,
            f1 = f_id
        )
        model.train()
        model
    end trainModel

    def hyperparameterTuning(data: Dataset, seed: Int = 42): BestParams =
        val (train, validation) = splitData(data, 0.20, seed)
        val (means, stds) = fitScaler(train.x)
        val xTrain = prepareX(train.x, means, stds)
        val yTrain = prepareY(train.y)
        val xVal = prepareX(validation.x, means, stds)
        val yVal = prepareY(validation.y)

        val hiddenNodeOptions = Array(8, 16, 32)
        val activationOptions = Array(("sigmoid", f_sigmoid), ("tanh", f_tanh))
        val learningRates = Array(0.01, 0.05, 0.1)
        val batchSizes = Array(32, 64)

        var bestMSE = Double.PositiveInfinity
        var bestNodes = 0
        var bestActivation = ""
        var bestLR = 0.0
        var bestBatch = 0

        for (activationName, activation) <- activationOptions do
            for nodes <- hiddenNodeOptions do
                for lr <- learningRates do
                    for batch <- batchSizes do
                        println(
                            s"Testing activation=$activationName, hiddenNodes=$nodes, " +
                            s"learningRate=$lr, batchSize=$batch"
                        )
                        val model = trainModel(xTrain, yTrain, nodes, activation, lr, batch)
                        val validationQoF = calculateQoF(yVal, model.predict(xVal))
                        println(f"Inner validation MSE = ${validationQoF.mse}%.5f")

                        if !validationQoF.mse.isNaN &&
                           !java.lang.Double.isInfinite(validationQoF.mse) &&
                           validationQoF.mse < bestMSE then
                            bestMSE = validationQoF.mse
                            bestNodes = nodes
                            bestActivation = activationName
                            bestLR = lr
                            bestBatch = batch
                        end if
                    end for
                end for
            end for
        end for

        require(bestNodes > 0, "Hyperparameter tuning did not produce a valid model.")
        val best = BestParams(bestNodes, bestActivation, bestLR, bestBatch, bestMSE)
        println(s"Selected hyperparameters: $best")
        best
    end hyperparameterTuning

    def activationFromName(name: String) =
        name match
            case "sigmoid" => f_sigmoid
            case "tanh"    => f_tanh
            case other     => throw new IllegalArgumentException(s"Unknown activation: $other")
    end activationFromName

    def runExperiment(
        name: String,
        path: String,
        target: String,
        seed: Int = 42
    ): BestParams =
        println(s"\n========== $name: 3L NEURAL NETWORK ==========")
        val data = loadDataset(path, target)
        println(s"Observations: ${data.y.length}")
        println(s"Predictors: ${data.x(0).length}")

        val (train, validation) = splitData(data, 0.20, seed)

        val best = hyperparameterTuning(train, seed + 1)

        val (means, stds) = fitScaler(train.x)
        val xTrain = prepareX(train.x, means, stds)
        val yTrain = prepareY(train.y)
        val xVal = prepareX(validation.x, means, stds)
        val yVal = prepareY(validation.y)

        val model = trainModel(
            xTrain,
            yTrain,
            best.hiddenNodes,
            activationFromName(best.activationName),
            best.learningRate,
            best.batchSize
        )

        val trainPred = model.predict(xTrain)
        val valPred = model.predict(xVal)
        val trainQoF = calculateQoF(yTrain, trainPred)
        val valQoF = calculateQoF(yVal, valPred)
        qofResults += ((name, trainQoF, valQoF))

        println(s"\n$name - Selected parameters")
        println(best)
        println(s"\n$name - Training QoF")
        println(f"R² = ${trainQoF.r2}%.5f")
        println(f"MSE = ${trainQoF.mse}%.5f")
        println(f"RMSE = ${trainQoF.rmse}%.5f")
        println(s"\n$name - Final validation QoF")
        println(f"R² = ${valQoF.r2}%.5f")
        println(f"MSE = ${valQoF.mse}%.5f")
        println(f"RMSE = ${valQoF.rmse}%.5f")

        model.opti.plotLoss(s"$name - 3L Training Loss")
        new Plot(yTrain(?, 0), trainPred(?, 0), null, s"$name - 3L Training: Actual vs Predicted", lines = false)
        new Plot(yVal(?, 0), valPred(?, 0), null, s"$name - 3L Validation: Actual vs Predicted", lines = false)

        best
    end runExperiment

    @main def runNeuralNet3L(): Unit =
        val airfoil = runExperiment(
            "Airfoil", "data/airfoil.csv", "scaled_sound_pressure_db", 42
        )
        val auto = runExperiment(
            "Auto MPG", "data/auto_mpg.csv", "mpg", 42
        )
        val concrete = runExperiment(
            "Concrete", "data/concrete.csv", "compressive_strength_mpa", 42
        )

        println("\n========== 3L HYPERPARAMETER TUNING RESULTS ==========")
        println("Dataset      Hidden nodes  Activation  Learning rate  Batch size  Inner validation MSE")
        for (name, best) <- Seq(
            ("Airfoil", airfoil),
            ("Auto MPG", auto),
            ("Concrete", concrete)
        ) do
            println(
                f"$name%-12s ${best.hiddenNodes}%-13d ${best.activationName}%-11s " +
                f"${best.learningRate}%-14.3f ${best.batchSize}%-11d ${best.validationMSE}%.5f"
            )
        end for

        println("\n========== 3L QUALITY OF FIT ==========")
        println(
            f"${"Dataset"}%-12s ${"Train R²"}%-12s ${"Validation R²"}%-16s " +
            f"${"Train MSE"}%-14s ${"Validation MSE"}%-17s ${"Train RMSE"}%-14s ${"Validation RMSE"}"
        )
        for (name, train, validation) <- qofResults do
            println(
                f"$name%-12s ${train.r2}%-12.5f ${validation.r2}%-16.5f " +
                f"${train.mse}%-14.5f ${validation.mse}%-17.5f " +
                f"${train.rmse}%-14.5f ${validation.rmse}%.5f"
            )
        end for
    end runNeuralNet3L

end NeuralNet3L
