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

from nemo_text_processing.text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Tamil ordinals, e.g.
        1வது  -> ordinal { integer: "ஒன்றாவது" }
        22வது -> ordinal { integer: "இருபத்திரண்டாவது" }
        2nd   -> ordinal { integer: "இரண்டாவது" }
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        suffixes_fst = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))
        exceptions   = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))

        limited_cardinal_graph = (
            cardinal.digit
            | cardinal.zero
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
        ).optimize()

        tamil_chars = pynini.union(*[chr(c) for c in range(0x0B80, 0x0C00)])
        sigma_mid   = pynini.closure(
            pynini.union(*[chr(c) for c in range(0x0B80, 0x0C00) if chr(c) != "ு"])
            | pynini.accep("ு") + (tamil_chars | pynini.accep(" "))
        )
        cardinal_to_ordinal = (limited_cardinal_graph @ (sigma_mid + pynini.cross("ு", "ாவது"))).optimize()

        graph = pynini.union(
            pynutil.add_weight(exceptions, -0.1),
            cardinal_to_ordinal + pynutil.delete(suffixes_fst)
        ).optimize()

        self.graph = graph.optimize()
        self.fst = self.add_tokens(pynutil.insert("integer: \"") + graph + pynutil.insert("\"")).optimize()