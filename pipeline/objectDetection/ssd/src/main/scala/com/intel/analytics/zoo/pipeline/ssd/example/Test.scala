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

package com.intel.analytics.zoo.pipeline.ssd.example

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.zoo.pipeline.common.{IOUtils, MeanAveragePrecision}
import com.intel.analytics.zoo.pipeline.ssd._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.ssd.model.PreProcessParam
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.models.ssd").setLevel(Level.INFO)

  case class PascolVocTestParam(folder: String = "",
    modelType: String = "vgg16",
    imageSet: String = "voc_2007_test",
    model: String = "",
    batch: Int = 8,
    className: String = "",
    resolution: Int = 300,
    nPartition: Int = 1)

  val parser = new OptionParser[PascolVocTestParam]("BigDL SSD Test") {
    head("BigDL SSD Test")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : VGG16 | PVANET")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]('i', "imageset")
      .text("imageset: voc_2007_test")
      .action((x, c) => c.copy(imageSet = x))
      .required()
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = x))
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batch = x))
    opt[String]("class")
      .text("class file")
      .action((x, c) => c.copy(className = x))
      .required()
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]) {
    parser.parse(args, PascolVocTestParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL SSD Test")
      val sc = new SparkContext(conf)
      Engine.init

      val classes = Source.fromFile(params.className).getLines().toArray
      val evaluator = new MeanAveragePrecision(true, normalized = true, classes = classes)
      val rdd = IOUtils.loadSeqFiles(params.nPartition, params.folder, sc)

      val model = Module.loadModule(params.model)
      println(s"load model done ${model.getName()}")

      val preprocess = if (params.modelType == "mobilenet") {
        PreProcessParam(params.batch, params.resolution,
          (127.5f, 127.5f, 127.5f), true, params.nPartition,
          (1 / 0.007843f, 1 / 0.007843f, 1 / 0.007843f))
      } else {
        PreProcessParam(params.batch, params.resolution,
          (123f, 117f, 104f), true, params.nPartition)
      }
      val validator = new Validator(model, preprocess, evaluator,
        useNormalized = true)

      validator.test(rdd)
    }
  }
}
