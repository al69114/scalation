package scalation
package modeling

import scalation.mathstat._

@main def Project2_Regression () : Unit =

    val datasets = Array(
        ("auto_mpg", "data/auto_mpg.csv", "mpg", Array("weight", "displacement")),
        ("concrete", "data/concrete.csv", "compressive_strength_mpa", Array("cement", "superplasticizer")),
        ("airfoil", "data/airfoil.csv", "scaled_sound_pressure_db",
            Array("frequency_hz", "suction_displacement_thickness_m"))
    )

    for (name, path, yLabel, topPredictors) <- datasets do

        banner(s"Project 2 - Regression - $name")

        val (xy, colNames) = MatrixD.loadH(path, fullPath = true)
        val n = xy.dim2 - 1
        val x = xy(?, 0 until n)
        val y = xy(?, n)
        val x_ = VectorD.one(x.dim) +^: x

        banner(
            s"$name: predictors = ${colNames.dropRight(1).mkString(", ")}, " +
            s"response = ${colNames.last}"
        )

        banner(s"$name Full Multiple Regression")

        val mod = new Regression(
            x_,
            y,
            colNames.dropRight(1) :+ "y"
        )

        mod.trainNtest()()

        banner(s"$name Fit Report")
        println(mod.report(mod.fit))
        println(mod.summary())

        for predictor <- topPredictors do

            banner(s"$name Simple Regression: $yLabel ~ $predictor")

            val predictorIndex = colNames.indexOf(predictor)

            if predictorIndex < 0 then
                println(s"ERROR: Could not find predictor '$predictor' in $name")
            else

                val xSimple = xy(?, predictorIndex)

                val values = Array.ofDim[Double](xSimple.dim * 2)

                for i <- 0 until xSimple.dim do
                    values(2 * i) = 1.0
                    values(2 * i + 1) = xSimple(i)

                val xSimple_ = MatrixD((xSimple.dim, 2), values*)

                println(s"xSimple dimensions = ${xSimple.dim}")
                println(s"xSimple_ dimensions = ${xSimple_.dim} x ${xSimple_.dim2}")

                val simpleMod = new Regression(
                    xSimple_,
                    y,
                    Array("Intercept", predictor)                
                    )

                simpleMod.trainNtest()()

                banner(s"$name - $predictor Fit Report")
                println(simpleMod.report(simpleMod.fit))
                println(simpleMod.summary())

                println(
                    s"\nSimple Regression: $yLabel = " +
                    s"intercept + coefficient * $predictor"
                )

        end for

    end for

end Project2_Regression