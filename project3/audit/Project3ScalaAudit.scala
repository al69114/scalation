package scalation.modeling.neuralnet
@main def project3ScalaAudit(): Unit =
    for (name,path,target) <- Seq(("Airfoil","data/airfoil.csv","scaled_sound_pressure_db"),("Auto MPG","data/auto_mpg.csv","mpg"),("Concrete","data/concrete.csv","compressive_strength_mpa")) do
        val d = NeuralNet2LProject.loadDataset(path,target)
        val (tr,va) = NeuralNet2LProject.splitData(d,0.2)
        val (mu,sd) = NeuralNet2LProject.fitScaler(tr.x)
        val m = NeuralNet2LProject.trainModel(NeuralNet2LProject.prepareX(tr.x,mu,sd),NeuralNet2LProject.prepareY(tr.y),0.3,16)
        println(s"AUDIT 2L $name optimizer=${m.opti.getClass.getSimpleName} train=${NeuralNet2LProject.calculateQoF(NeuralNet2LProject.prepareY(tr.y),m.predict(NeuralNet2LProject.prepareX(tr.x,mu,sd)))} validation=${NeuralNet2LProject.calculateQoF(NeuralNet2LProject.prepareY(va.y),m.predict(NeuralNet2LProject.prepareX(va.x,mu,sd)))}")
        val d3 = NeuralNet3L.loadDataset(path,target)
        val (t3,v3) = NeuralNet3L.splitData(d3,0.2)
        val (mu3,sd3) = NeuralNet3L.fitScaler(t3.x)
        val (nodes,act,lr,bs) = name match
            case "Airfoil" => (32,"tanh",0.05,64)
            case "Auto MPG" => (8,"sigmoid",0.05,32)
            case _ => (16,"tanh",0.01,64)
        val n = NeuralNet3L.trainModel(NeuralNet3L.prepareX(t3.x,mu3,sd3),NeuralNet3L.prepareY(t3.y),nodes,NeuralNet3L.activationFromName(act),lr,bs)
        println(s"AUDIT 3L $name optimizer=${n.opti.getClass.getSimpleName} train=${NeuralNet3L.calculateQoF(NeuralNet3L.prepareY(t3.y),n.predict(NeuralNet3L.prepareX(t3.x,mu3,sd3)))} validation=${NeuralNet3L.calculateQoF(NeuralNet3L.prepareY(v3.y),n.predict(NeuralNet3L.prepareX(v3.x,mu3,sd3)))}")
