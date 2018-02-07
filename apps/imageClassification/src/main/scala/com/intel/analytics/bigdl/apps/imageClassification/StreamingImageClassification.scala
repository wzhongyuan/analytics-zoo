/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.apps.imageClassification

import java.util
import java.util.Random

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{LocalPredictor, Predictor}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, ImageFrame, ImageFrameToSample, LocalImageFrame, MatToTensor}
import com.intel.analytics.bigdl.utils.{Engine, File}
import org.apache.storm.{Config, LocalCluster, StormSubmitter}
import org.apache.storm.spout.SpoutOutputCollector
import org.apache.storm.task.TopologyContext
import org.apache.storm.topology.{BasicOutputCollector, OutputFieldsDeclarer, TopologyBuilder}
import org.apache.storm.topology.base.{BaseBasicBolt, BaseRichSpout}
import org.apache.storm.tuple.{Fields, Tuple, Values}
import scopt.OptionParser

import scala.io.Source

/**
 * An example to integrate BigDL into Storm streaming using InceptionV1
 */

object StreamingImageClassification {

  /**
   *  @param imageFolder  where to put sample images
   *  @param model   model path
   *  @param threshold the predicted possibility of target should be >= threahold
   *  @param target  target category keyword
   *  @param resultFolder where to put filtered images
   *  @param localMode if run in Storm local or distributed mode
   */
  case class ClassificationParam(imageFolder: String = "",
                                 model: String = "",
                                 threshold: Double = 0.0f,
                                 target: String = "",
                                 labelPath: String = "",
                                 resultFolder: String = "",
                                 localMode: Boolean = true)

  val parser = new OptionParser[ClassificationParam]("Streaming ImageClassification demo") {
    head("Image Classification with BigDL and Storm")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("model")
      .text("BigDL model path")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Double]("threshold")
      .text("Threshold for filter")
      .action((x, c) => c.copy(threshold = x))
      .required()
    opt[String]("target")
      .text("target keywork to filter out images")
      .action((x, c) => c.copy(target = x))
      .required()
    opt[String]("labelPath")
      .text("labelPath")
      .action((x, c) => c.copy(labelPath = x))
      .required()
    opt[String]("resultFolder")
      .text("result folder to put filtered images")
      .action((x, c) => c.copy(resultFolder = x))
      .required()
    opt[Boolean]("localMode")
      .text("run in local model or not")
      .action((x, c) => c.copy(localMode = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, ClassificationParam()).foreach { param =>
      val imgFolder = param.imageFolder
      val modelPath = param.model
      val threshold = param.threshold
      val target = param.target
      val labelPath = param.labelPath
      val resultFolder = param.resultFolder
      val localMode = param.localMode

      val conf = new Config

      val builder = new TopologyBuilder
      builder.setSpout("spout", new ImageSpout(imgFolder))

      builder.setBolt("predict", new ImagePredictor(modelPath)).
        shuffleGrouping("spout")

      builder.setBolt("labelOutput", new LabelResult(labelPath)).
        shuffleGrouping("predict")

      builder.setBolt("filter", new ImageFilter(resultFolder, threshold.toFloat, target)).
        shuffleGrouping("labelOutput")
      conf.setDebug(true)
      conf.setNumWorkers(4)
      if (localMode) {
        val cluster = new LocalCluster
        cluster.submitTopology("StreamingClassification", conf, builder.createTopology)
      } else {
        StormSubmitter.submitTopologyWithProgressBar("StreamingClassification", conf,
          builder.createTopology)
      }
    }
  }
}

// A Spout implementation to simulate continously feeding data

class ImageSpout(val path: String) extends BaseRichSpout{

  private var collector : SpoutOutputCollector = null

  var imageFrame : ImageFrame = ImageFrame.read(path)

  val images = imageFrame.asInstanceOf[LocalImageFrame].array

  private var rand: Random = null

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }

  override def nextTuple(): Unit = {
    Thread.sleep(1000)
    val image = images(rand.nextInt(images.length))
    val imageFram = ImageFrame.array(Array(image))
    collector.emit(new Values(imageFram))
  }

  override def open(map: util.Map[_, _],
                    topologyContext: TopologyContext,
                    spoutOutputCollector: SpoutOutputCollector): Unit = {
    collector = spoutOutputCollector
    rand = new Random
  }
}

class ImagePredictor(val modelPath: String) extends BaseBasicBolt {

  private var model : AbstractModule[Activity, Activity, Float] = null
  private var predictor : LocalPredictor[Float] = null
  val transformer: FeatureTransformer = Resize(256, 256) ->
    CenterCrop(224, 224) -> ChannelNormalize(123, 117, 104) ->
    MatToTensor[Float]() -> ImageFrameToSample[Float]()
  override def execute(tuple: Tuple, basicOutputCollector: BasicOutputCollector): Unit = {
    val imageFrame = tuple.getValue(0).asInstanceOf[ImageFrame].toLocal
    imageFrame -> transformer
    basicOutputCollector.emit(new Values(predictor.predictImage(imageFrame)))
  }

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }

  override def prepare(stormConf: util.Map[_, _], context: TopologyContext): Unit = {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", 1.toString)
    Engine.init
    model = Module.loadModule[Float](modelPath)
    predictor = LocalPredictor[Float](model)
  }
}

class LabelResult(val labelPath: String) extends BaseBasicBolt {

  private var labelMap : Map[Int, String] = null
  override def execute(tuple: Tuple, basicOutputCollector: BasicOutputCollector): Unit = {
    val imageFrame = tuple.getValue(0).asInstanceOf[ImageFrame]
    imageFrame.toLocal().array.foreach(imageFeature => {
      val predictOutput = imageFeature[Tensor[Float]](ImageFeature.predict)
      val total = predictOutput.nElement()
      val start = predictOutput.storageOffset() - 1
      val end = predictOutput.storageOffset() - 1 + predictOutput.nElement()
      val clsNo = end - start
      val sortedResult = predictOutput.storage().array().slice(start, end).
        zipWithIndex.sortWith(_._1 > _._1).toList.toArray

      val classes: Array[String] = new Array[String](clsNo)
      val probilities  : Array[Float] = new Array[Float](clsNo)

      var index = 0;

      while (index < clsNo) {
        val clsName = labelMap(sortedResult(index)._2)
        val prob = sortedResult(index)._1
        classes(index) = clsName
        probilities(index) = prob
        index += 1
      }

      imageFeature("clses") = classes
      imageFeature("probs") = probilities
    })
   basicOutputCollector.emit(new Values(imageFrame))
  }

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }

  override def prepare(stormConf: util.Map[_, _], context: TopologyContext): Unit = {
    labelMap = Source.fromFile(labelPath).getLines().zipWithIndex.map(x => (x._2, x._1)).toMap
  }
}

class ImageFilter(val targetFolder: String, val threshold: Float,
                  val keyWord: String) extends BaseBasicBolt {
  override def execute(tuple: Tuple, basicOutputCollector: BasicOutputCollector): Unit = {
    val images = tuple.getValue(0).asInstanceOf[ImageFrame].toLocal().array
    images.foreach(imageFeature => {
      val clsses = imageFeature("clses").asInstanceOf[Array[String]]
      val probs = imageFeature("probs").asInstanceOf[Array[Float]]
      var found: Boolean = false
      var index = 0
      while (index < probs.length && probs(index) >= threshold && !found) {
        if (clsses(index).contains(keyWord)) {
          found = true
        }
        index += 1
      }
      if (found) {
        val uri = imageFeature(ImageFeature.uri).asInstanceOf[String]
        val fileName = uri.substring(uri.lastIndexOf("/") + 1, uri.length)
        val rawImage = imageFeature(ImageFeature.bytes).asInstanceOf[Array[Byte]]
        val path = s"$targetFolder/${System.currentTimeMillis()}_${fileName}"
        File.saveBytes(rawImage, path, true)
      }
    })
  }

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }
}