|      Name     |                                                  Paper                                                 |                          Repository                          |            Model(s)           |                                      Model(s) repo                                     | Pretrained |           Base model(s)          |                                        Task(s)                                       |       Language      |
|:-------------:|:------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|:-----------------------------:|:--------------------------------------------------------------------------------------:|:----------:|:--------------------------------:|:------------------------------------------------------------------------------------:|:-------------------:|
| Tufano et al. | An Empirical Investigation into Learning Bug-Fixing Patches in the Wild via Neural Machine Translation |         https://sites.google.com/view/learning-fixes         | RNN Encoder-Decoder (seq2seq) |                                            -                                           |     NO     |                 -                |                                           -                                          |         Java        |
|    CoCoNut    |      CoCoNuT: combining context-aware neural translation models using ensemble for program repair      |          https://github.com/lin-tan/CoCoNut-Artifact         |       Ensemble NMT & CCN      |                                            -                                           |     NO     |                 -                |                                           -                                          | Java, C, Python, JS |
|  Tian et al.  |  Evaluating Representation Learning of Code Changes for Predicting Patch Correctness in Program Repair |        https://github.com/TruX-DTF/DL4PatchCorrectness       |          Transformer          |  https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip  |     Yes    |               BERT               |                             Predictiong Patch Correctness                            |         Java        |
|    Recoder    |                         A Syntax-Guided Edit Decoder for Neural Program Repair                         |               https://github.com/pkuzqh/Recoder              |        Encoder-Decoder        |                                            -                                           |     No     |                 -                |                                           -                                          |         Java        |
|      TFix     |                   TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer                  |                https://github.com/eth-sri/TFix               |          Transformer          |   https://drive.google.com/file/d/1CtfnYaVf-q6FZP5CUM4Wh7ofpp8b9ajW/view?usp=sharing   |     Yes    |              T5large             |                                      Code fixing                                     |          JS         |
|    CodeBERT   |                   Applying CodeBERT for Automated Program Repair of Java Simple Bugs                   |    https://github.com/EhsanMashhadi/MSR2021-ProgramRepair    |          Transformer          |                                            -                                           |     Yes    |             CodeBERT             |                                      Code fixing                                     |         Java        |
|       T5      |             Studying the usage of text-to-text transfer transformer for code-related tasks             | https://github.com/antonio-mastropaolo/TransferLearning4Code |           Tranformer          |        https://drive.google.com/drive/folders/1R23fXWC8YPz3SgLDp-BxcLXAQ1exh4Vh        |     Yes    |              T5small             | Code fixing, inject code mutants, generate assert statements, generate code comments |         Java        |
|     GPT-2     |            Towards JavaScript program repair with Generative Pre-trained Transformer (GPT-2)           |            https://github.com/AAI-USZ/APR22-JS-GPT           |          Transformer          |                                            -                                           |     Yes    |               GPT-2              |                                      Code fixing                                     |          JS         |
|     CodiT5    |                    CoditT5: Pretraining for Source Code and Natural Language Editing                   |        https://github.com/EngineeringSoftware/CoditT5        |          Transformer          |                                                                                        |     Yes    |       PLBART,GPT-2, CodeT5       |                  Comment updating, bug fixing, automated code review                 |         Java        |
|    SYNSHINE   |                               SYNSHINE: improved fixing of Syntax Errors                               |        https://zenodo.org/record/4572390#.ZBCxinbMJD9        |          Transformer          |                     https://zenodo.org/record/4572390#.ZBCxinbMJD9                     |     Yes    |           BERT, RoBERTA          |                   Fill in masked out tokens, predict next sentence                   |                     |
|       -       |                       Impact of Code Language Models on Automated Program Repair                       |                https://github.com/lin-tan/clm                |           Tranformer          | https://zenodo.org/record/7559244#.ZBC_w3bMJD8, https://doi.org/10.5281/zenodo.7559277 |     Yes    | PLBART, CodeT5, CodeGen, InCoder |                                      Bug fixing                                      |         Java        |
|       -       |             Automating Code-Related Tasks Through Transformers: The Impact of Pre-training             |     https://github.com/RosaliaTufano/impact_pre-training     |          Transformer          |                     https://zenodo.org/record/7078746#.ZBDWRHbMJD9                     |     Yes    |              T5small             |  Injected mutant fixing, masked, next sentence prediction, replaced model detection  |         Java        |