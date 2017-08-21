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

package com.intel.analytics.zoo.feature.core.image

import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.feature.core.util.MatWrapper

class Contrast(deltaLow: Double, deltaHigh: Double)
  extends FeatureTransformer {
  override def transform(feature: Feature): Unit = {
    Contrast.transform(feature.inputMat(), feature.inputMat(), RNG.uniform(deltaLow, deltaHigh))
  }
}

object Contrast {
  def apply(deltaLow: Double, deltaHigh: Double): Contrast = new Contrast(deltaLow, deltaHigh)

  def transform(input: MatWrapper, output: MatWrapper, delta: Double): MatWrapper = {
    if (Math.abs(delta - 1) > 1e-3) {
      input.convertTo(output, -1, delta, 0)
    } else {
      if (input != output) input.copyTo(output)
    }
    output
  }
}