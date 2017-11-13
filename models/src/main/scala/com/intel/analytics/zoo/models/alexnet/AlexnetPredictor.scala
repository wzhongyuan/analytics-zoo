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

package com.intel.analytics.zoo.models.alexnet

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.zoo.models.{Predictor}
import com.intel.analytics.zoo.models.dataset._
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{CenterCrop, PixelNormalizer, Resize}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

@SerialVersionUID(6044110396967995592L)
class AlexnetPredictor(modelPath : String, meanPath : String) extends Predictor with Serializable{

  model = ModuleLoader.loadFromFile[Float](modelPath).evaluate()

  val mean : Array[Float] = createMean(meanPath)

  val transformer = ImageToBytes() -> BytesToMat() -> Resize(256, 256) ->
    PixelNormalizer(mean) -> CenterCrop(227, 227) -> MateToSample(false)


  override def predictLocal(path : String, topNum : Int,
                            preprocessor: Transformer[String, ImageSample] = transformer)
  : PredictResult = {
    doPredictLocal(path, topNum, preprocessor)
  }

  override def predictDistributed(paths : RDD[String], topNum : Int,
                                  preprocessor: Transformer[String, ImageSample] = transformer):
    RDD[PredictResult] = {
    doPredictDistributed(paths, topNum, preprocessor)
  }

  private def createMean(meanFile : String) : Array[Float] = {
    val lines = Files.readAllLines(Paths.get(meanFile), StandardCharsets.UTF_8)
    val array = new Array[Float](lines.size)
    lines.toArray.zipWithIndex.foreach {
      x => {
        array(x._2) = x._1.toString.toFloat
      }
    }
    array
  }
}

object AlexnetPredictor{

  def apply(modelPath: String, meanPath : String): AlexnetPredictor =
    new AlexnetPredictor(modelPath, meanPath)
}