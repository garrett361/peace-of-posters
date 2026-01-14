#import "../lib.typ" as pop

// #set page("a0", margin: 1cm)
#set page(width: 121.92cm, height: 91.44cm, margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.ibm)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#let GG(content) = text(fill: red)[GG: #content]

#pop.title-box(
  "Context-Parallel Mamba2: Scaling to 1M+ Tokens",
  authors: "Garrett Goon",
  // institutes: "IBM Research",
  // keywords: "Peace, Dove, Poster, Science",
  logo: image("figs/ibm_logo_black_back-cropped.svg", width: 400pt),
  title-size: 130pt,
  authors-size: 90pt,
)

#columns(
  3,
  [

    #pop.column-box(heading: "Mamba2: A Linear Attention Layer")[

      The quadratic $cal(O)( mono("seqlen")^( 2 ) )$ scaling of Transformers attention has
      increasingly become a bottleneck as context lengths have ballooned with the advent of reasoning
      models and rising prevalence of modalities such as video and audio.

      #set align(center)
      #table(
        columns: 3,
        inset: (x: 5pt, y: 20pt),
        [Architecture], [*Prefill/Train*], [*Prefill/Train*],
        [Attention],
        [$cal(O)( mono("seqlen")^2 )$],
        [$cal(O)( mono("seqlen") )$],

        [Mamba2], [$cal(O)( mono("seqlen") )$], [$cal(O)( 1 )$],
      )
      #set align(left)

      Mamba2 @dao2024transformersssmsgeneralizedmodels is an alternative information propagation
      algorithm belonging to the steadily growing class of _linear_ $cal(O)( mono("seqlen") )$
      attention mechanisms @fla. Its GPU-aware design enables hardware utilization comparable to
      quadratic attention during training, and reduces the decoding time and cache-space from $cal(O)(
    mono("seqlen") )$ to $cal(O)( 1 )$. See @glorioso2024zambacompact7bssm @bamba
      @nvidia2025nemotronhfamilyaccurateefficient @granite for a partial list of models which utilize
      Mamba2
    ]



    #pop.column-box(heading: "Mamba2 Architecture (Simplified)")[
      Mamba2 relies on two central mechanisms for propagating information:
      + Short 1D causal convolutions @causalconv1d
      + A gated recursion relation
      Giving inputs $x_( s d ) in bb(R)^( mono("seqlen") times mono("d_model") ) $, the Mamba2 outputs are schematically
      $
        z_( s d ) ~ mono("gated_recursion")(mono("causal_conv1d")(x_( s d )))
      $
      Both components require adaptation in the CP implementation. 

      == Causal Convolutions

      The causal convolution @causalconv1d is a depthwise, 1D convolution along the sequence dimension with a short
      filter, typically of width $K=4$ :
      $
        z_( s d ) = sum_( k= 0 )^( K ) W_( k d ) x_( (s-k) d ) space .
      $

      == Gated Recursion Relations

      The central elements in the Mamba2 recursion relation are of the form:
      $
        z_( s d ) = e^( -A_( s ) ) z_( (s-1)d ) + Delta_( s ) x_( s d )
      $
      where the data-dependent $A_( s ) , Delta_( s ) >= 0$ control the deletion and addition of information to the
      state $z_( s d )$. While the complete tensor $z_( s d )$ can be constructed in $cal(O)(
    mono("seqlen") )$ by solving the recursion in the naive manner, such an approach is suboptimal
      in practice as it cannot leverage GPU tensor cores. For this reason, the recursion is
      solved using a chunked, matmul-based strategy which has inferior theoretical scaling, but
      superior in-practice wall time @dao2024transformersssmsgeneralizedmodels.

      #figure(
      )[
        #image("figs/mamba_scan_and_conv.png", width: 60%)
      ]

    ]

    #pop.column-box(heading: "Context Parallelism")[

      Leveraging Mamba2's long-context scaling advantages requires long-context training which
poses engineering challenges as $mono("activation_mem") prop mono("seqlen") $ .

      Context-parallelism (CP), in which sequences are sharded along the sequence
      dimension, is a natural and scalable approach to long-sequence training.  

    // #GG[Should probably mention ring attn somewhere]


      #figure(
        caption: [
          Context Parallel Sharding
        ],
      )[
        #image("figs/simple_cp_chunking.png", width: 100%)
      ]

    ]

    #pop.column-box(heading: "CP Convolutions")[

      #figure(
        caption: [
          Context Parallel Causal Convolution
        ],
      )[
        #image("figs/cp_causal_conv.png", width: 90%)
      ]

      Only minimal modifications required for CP causal convolutions. Each GPU handles $C =
    mono("seqlen") \/ mono("cp_degree")$ tokens, and only the outputs for the first $K-1 << C$
      tokens require communication for computational correctness.

      An efficient algorithm is as follows:
      + CP rank $mono("r")$ asynchronously passes its final $K-1$ tokens to rank $mono("r+1")$
      + Concurrently, ranks convolve their local tokens: `z = causal_conv1d(x)`
      + The leading outputs are corrected via the received tokens: \ `z[:K-1] = causal_conv1d(cat(x_recv, x[:K-1]))`


    ]

    #pop.column-box(heading: "CP Rescursion: State Passing")[
      A straightforward CP implementation of Mamba2's recursion step comes from passing state across
CP boundaries. This proceeds as follows:

      + Rank $r$ solves the recursion locally and passes `z[-1]` to rank `r + 1`.
      + Rank $r+1$ waits on the passed state, then solves recursion locally. 

      The causal Mamba2 dependencies result in $cal(O)( mono("seqlen") )$ exposed communication. However,
pipelining can be use to amortize this cost.


      #figure(
      )[
        #image("figs/serial_cp_pipelining.png", width: 50%)
      ]

    ]

    #pop.column-box(heading: "CP Recursion: Compute-Then-Correct")[

      An alternative implementation which can have better strong-scaling properties is as follows:

      Alternatively, a strategy similar to the CP convolution algorithm is also possible, in which we
      compute incorrect outputs with locally available tensors, and then correct the results via
      communication. GPUs never idle with this strategy.

      In order to describe this strategy, we trade the global sequence index $s$ for the pair of indices $r, c$ with $r in {0, ..., mono("cp_degree") - 1}$ indexing
      the CP rank and $c in {0, ..., mono("seqlen") \/ mono("cp_degree") - 1 } $ indexing the chunked
      sequence position. A valid CP implementation is as follows:
      + Every rank computes $Sigma_( r ) = sum_( c ) A_( r c )$: the sum of local gate values
      + Rank $r$ asynchronously sends $Sigma_( r )$ to CP ranks $r\' > r$
      + Every rank solves the recursion relation with its local inputs $x_( r c d )$ and trivial initial state, producing (incorrect) final states $y^( "final" )_(r d )$
      + Rank $r$ sends $y^( "final" )_( r d )$ to CP ranks $r\' > r$
      + Rank $r$ computes its corrected initial state via $x_( r d )^( "initial" ) = sum_( r\' < r ) exp(Sigma_( r - 1 ) + ... + Sigma_( r\' + 1 ))y^( "final" )_( r\' d )$
      + Every rank re-solves the recursion relation with its now-corrected initial states, yielding the correct outputs $z_( r c d )$

      This strategy requires $cal(O)( mono("seqlen") \/ mono("cp_degree") )$ additional compute and has $cal(O)( mono("cp_degree") )$ exposed communication.

    ]

    // These properties will be given to the function which is responsible for creating the heading
    #let hba = pop.uni-fr.heading-box-args
    #hba.insert(
      "stroke",
      (paint: gradient.linear(green, red, blue), thickness: 10pt),
    )

    // and these are for the body.
    #let bba = pop.uni-fr.body-box-args
    #bba.insert("inset", 30pt)
    #bba.insert(
      "stroke",
      (paint: gradient.linear(green, red, blue), thickness: 10pt),
    )


    #pop.column-box()[
      #bibliography("bibliography.bib", title: "References")
    ],

  ],
)
