/---------------------------------------------------------------------------/
/                   Spectra NET for classifying crops                       /
/                                                                           /
/   This project is a simple experiment with the                            /
/   paper proposed by Zeit AI on predicting soy                             /
/   beans contents with a CNN regressor.                                    /
/   my approach was basically the same as in                                /
/   the paper, only differing by the amount of                              /
/   neurons on the dense layers, weights                                    /
/   initialization, learning rate and the fact                              /
/   that i did not apply data augmentation                                  /
/                                                                           /
/   Results:                                                                /
/   The network performed really well achieving                             /
/   a precision of 82% (on average) and a whopping                          /
/   84% recall (on average as well)                                         /
/                                                                           /
/   The paper can be found at:                                              /
/   https://www.scitepress.org/publishedPapers/2024/126976/pdf/index.html   /
/ --------------------------------------------------------------------------/