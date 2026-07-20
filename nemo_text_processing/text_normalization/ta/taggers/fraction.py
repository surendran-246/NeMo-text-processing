# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ta.graph_utils import (
    TA_ARAI,
    TA_MUKKAAL,
    NEMO_SPACE,
    GraphFst,
)

TA_ONE_HALF = "௧/௨"  # 1/2
TA_ONE_QUARTER = "௧/௪"  # 1/4
TA_THREE_QUARTERS = "௩/௪"  # 3/4


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "௨௩ ௪/௬" ->
    fraction { integer: "இருபத்து மூன்று" numerator: "நான்கு" denominator: "ஆறு"}
    "௪/௬" ->
    fraction { numerator: "நான்கு" denominator: "ஆறு"}


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + pynutil.insert(NEMO_SPACE), 0, 1
        )
        self.integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.numerator = (
            pynutil.insert("numerator: \"")
            + cardinal_graph
            + pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), "\"")
            + pynutil.insert(NEMO_SPACE)
        )
        self.denominator = pynutil.insert("denominator: \"") + cardinal_graph + pynutil.insert("\"")

        # ---- Mixed forms: "<number> மற்றும் <word>" (e.g. இருபது மற்றும் கால்) ----
        onrukaal_numbers = cardinal_graph + pynini.cross(NEMO_SPACE + TA_ONE_QUARTER, "")
        onrukaal_graph = (
            onrukaal_numbers
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("மற்றும்")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("கால்")
        )

        arai_numbers = cardinal_graph + pynini.cross(NEMO_SPACE + TA_ONE_HALF, "")
        arai_graph = (
            arai_numbers
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("மற்றும்")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert(TA_ARAI)
        )

        mukkaal_numbers = cardinal_graph + pynini.cross(NEMO_SPACE + TA_THREE_QUARTERS, "")
        mukkaal_graph = (
            mukkaal_numbers
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("மற்றும்")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert(TA_MUKKAAL)
        )

        # Bare forms: standalone fraction only, no integer part (e.g. கால், அரை, முக்கால்) 
        bare_kaal_graph = pynini.cross(TA_ONE_QUARTER, "கால்")
        bare_arai_graph = pynini.cross(TA_ONE_HALF, TA_ARAI)
        bare_mukkaal_graph = pynini.cross(TA_THREE_QUARTERS, TA_MUKKAAL)

        graph_onrukaal = (
            pynutil.insert("morphosyntactic_features: \"")
            + onrukaal_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_arai = (
            pynutil.insert("morphosyntactic_features: \"")
            + arai_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_mukkaal = (
            pynutil.insert("morphosyntactic_features: \"")
            + mukkaal_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_bare_kaal = (
            pynutil.insert("morphosyntactic_features: \"")
            + bare_kaal_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_bare_arai = (
            pynutil.insert("morphosyntactic_features: \"")
            + bare_arai_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_bare_mukkaal = (
            pynutil.insert("morphosyntactic_features: \"")
            + bare_mukkaal_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        final_graph = (
            self.optional_graph_negative
            + pynini.closure(self.integer + pynini.accep(NEMO_SPACE), 0, 1)
            + self.numerator
            + self.denominator
        )

        weighted_graph = (
            final_graph
            | pynutil.add_weight(graph_onrukaal, -0.1)
            | pynutil.add_weight(graph_arai, -0.1)
            | pynutil.add_weight(graph_mukkaal, -0.2)
            | pynutil.add_weight(graph_bare_kaal, -0.3)
            | pynutil.add_weight(graph_bare_arai, -0.3)
            | pynutil.add_weight(graph_bare_mukkaal, -0.3)
        )

        self.graph = weighted_graph

        graph = self.graph
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()