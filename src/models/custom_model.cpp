#include "model.h"

unsigned char custom_model_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x1c, 0x00, 0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x90, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x14, 0x03, 0x00, 0x00,
  0x24, 0x03, 0x00, 0x00, 0xf8, 0x08, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x73, 0x65, 0x72, 0x76, 0x69, 0x6e, 0x67, 0x5f,
  0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xc0, 0xff, 0xff, 0xff, 0x09, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x64, 0x65, 0x6e, 0x73,
  0x65, 0x5f, 0x32, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x86, 0xfd, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x6e, 0x5f,
  0x72, 0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x5f, 0x76, 0x65, 0x72, 0x73,
  0x69, 0x6f, 0x6e, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x50, 0x02, 0x00, 0x00,
  0x48, 0x02, 0x00, 0x00, 0x28, 0x02, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x00,
  0xc0, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x3c, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x06, 0xfe, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x31, 0x2e, 0x31, 0x34,
  0x2e, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x84, 0xfd, 0xff, 0xff, 0x88, 0xfd, 0xff, 0xff, 0x8c, 0xfd, 0xff, 0xff,
  0x2e, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xdb, 0xfb, 0xff, 0xff, 0x3e, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x16, 0x9c, 0x17, 0xd1, 0xf4, 0x43, 0xbe, 0x18,
  0x7f, 0x59, 0x25, 0xdb, 0x2d, 0xca, 0xe2, 0xd2, 0x5a, 0xfe, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x26, 0xfc, 0xff, 0xff,
  0xf1, 0x0c, 0x00, 0x00, 0xcf, 0xfc, 0xff, 0xff, 0x00, 0x06, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xee, 0xf9, 0xff, 0xff, 0x9f, 0x01, 0x00, 0x00,
  0x17, 0xff, 0xff, 0xff, 0xfa, 0x0b, 0x00, 0x00, 0xe8, 0x0f, 0x00, 0x00,
  0x48, 0xfc, 0xff, 0xff, 0x04, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x55, 0x02, 0x00, 0x00, 0xfb, 0x03, 0x00, 0x00, 0x45, 0x04, 0x00, 0x00,
  0xa6, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0xe1, 0x05, 0xf8, 0x25, 0xe4, 0x2b, 0x25, 0xf3, 0xff, 0x01, 0xf3, 0x26,
  0xda, 0x48, 0x15, 0x28, 0xf0, 0xd2, 0x1c, 0xf8, 0xf5, 0x2e, 0xc4, 0x1e,
  0xd2, 0xe9, 0x1e, 0xb6, 0x51, 0xea, 0xb8, 0x11, 0xdc, 0xfc, 0xe2, 0x2e,
  0x07, 0x28, 0x24, 0xe9, 0x27, 0x08, 0x05, 0xe6, 0xee, 0x44, 0x28, 0x2c,
  0xd7, 0x1a, 0xd6, 0xd3, 0xee, 0xd4, 0x18, 0x14, 0xe1, 0xdb, 0xd2, 0xdd,
  0x35, 0xf2, 0xa8, 0x10, 0x01, 0xf5, 0x18, 0xf8, 0x2e, 0xd1, 0x23, 0x1a,
  0xf1, 0x11, 0x27, 0xf1, 0xf1, 0xe8, 0xff, 0xec, 0xf9, 0x0e, 0x1a, 0x0e,
  0x19, 0xf4, 0xfc, 0xf4, 0x1d, 0x42, 0x25, 0x1a, 0xa4, 0x44, 0xf4, 0x2d,
  0x07, 0x19, 0xd9, 0x05, 0xe8, 0xf7, 0x1c, 0xe5, 0xfa, 0xfd, 0x22, 0xf0,
  0x4e, 0xe2, 0x2e, 0xec, 0xea, 0xda, 0xfd, 0xd4, 0xed, 0xec, 0xd8, 0x2a,
  0xd7, 0x19, 0x00, 0xdc, 0x1e, 0x23, 0xfb, 0x17, 0x0c, 0x0f, 0xc9, 0x08,
  0xf2, 0x17, 0x0c, 0xe2, 0x1c, 0xd0, 0xfa, 0xb5, 0x5c, 0xd9, 0xf1, 0x2e,
  0x20, 0xf1, 0xdf, 0xff, 0xdd, 0x1b, 0xfb, 0xab, 0x24, 0x81, 0xda, 0xc4,
  0x64, 0xf2, 0xe0, 0x1b, 0x16, 0xe5, 0x15, 0x16, 0xe1, 0x09, 0x26, 0x28,
  0x1e, 0x22, 0xe9, 0x32, 0xd4, 0x11, 0xe9, 0xda, 0x0f, 0x10, 0x16, 0xfc,
  0xd8, 0xeb, 0x0d, 0xd8, 0x1d, 0xe9, 0xd7, 0xe4, 0x36, 0xf2, 0xf1, 0xe9,
  0xd8, 0x13, 0xff, 0xe8, 0xe6, 0xd1, 0xff, 0x20, 0xe3, 0x0c, 0xf9, 0xe7,
  0xfc, 0xf0, 0xd8, 0xeb, 0xdd, 0xed, 0xec, 0x06, 0x10, 0xdf, 0xf6, 0xd8,
  0x18, 0x08, 0x24, 0xfe, 0x23, 0xbd, 0x31, 0x0e, 0x2b, 0x14, 0x17, 0xd9,
  0xf6, 0xdb, 0x21, 0x0c, 0x15, 0xe1, 0xd9, 0x24, 0x12, 0xe3, 0xf5, 0x1d,
  0x16, 0x19, 0x00, 0xd1, 0xe5, 0x29, 0x2b, 0xc4, 0x27, 0xbf, 0xe2, 0xd0,
  0x69, 0xe4, 0x29, 0xf6, 0xb2, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x05, 0xef, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x74, 0x01, 0x00, 0x00, 0x0a, 0x08, 0x00, 0x00, 0x4c, 0xe9, 0xff, 0xff,
  0xc9, 0x09, 0x00, 0x00, 0x43, 0xe6, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x98, 0xea, 0xff, 0xff, 0x46, 0x1d, 0x00, 0x00, 0xcc, 0xed, 0xff, 0xff,
  0x65, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0xac, 0xbe, 0x5d, 0x81, 0xee, 0x57, 0x39, 0x38,
  0x7f, 0x44, 0x8b, 0x43, 0xf9, 0x15, 0x6b, 0xf0, 0x84, 0xff, 0xff, 0xff,
  0x88, 0xff, 0xff, 0xff, 0x0f, 0x00, 0x00, 0x00, 0x4d, 0x4c, 0x49, 0x52,
  0x20, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0xdc, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x00, 0xe4, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x84, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x96, 0xff, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x08, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0xca, 0xff, 0xff, 0xff,
  0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x10, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0xba, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x0e, 0x00, 0x16, 0x00, 0x00, 0x00, 0x10, 0x00, 0x0c, 0x00,
  0x0b, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x08, 0x18, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x3c, 0x04, 0x00, 0x00, 0xc4, 0x03, 0x00, 0x00, 0x60, 0x03, 0x00, 0x00,
  0xf4, 0x02, 0x00, 0x00, 0x98, 0x02, 0x00, 0x00, 0x2c, 0x02, 0x00, 0x00,
  0xd0, 0x01, 0x00, 0x00, 0x2c, 0x01, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0xfc, 0xff, 0xff, 0x18, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x54, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0xec, 0xfb, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0xfb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x2f, 0x05, 0xfd, 0x3b, 0x19, 0x00, 0x00, 0x00,
  0x53, 0x74, 0x61, 0x74, 0x65, 0x66, 0x75, 0x6c, 0x50, 0x61, 0x72, 0x74,
  0x69, 0x74, 0x69, 0x6f, 0x6e, 0x65, 0x64, 0x43, 0x61, 0x6c, 0x6c, 0x3a,
  0x30, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x78, 0xfc, 0xff, 0xff, 0x18, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x84, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x64, 0xfc, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x0c, 0xbb, 0x4b, 0x3c, 0x4c, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75,
  0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65,
  0x5f, 0x31, 0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x3b, 0x73, 0x65,
  0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e,
  0x73, 0x65, 0x5f, 0x31, 0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x73, 0x65,
  0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e,
  0x73, 0x65, 0x5f, 0x31, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64,
  0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x20, 0xfd, 0xff, 0xff, 0x18, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x7c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x0c, 0xfd, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x46, 0x6c, 0x7c, 0x3c, 0x46, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75,
  0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65,
  0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x3b, 0x73, 0x65, 0x71, 0x75,
  0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65,
  0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e,
  0x74, 0x69, 0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x2f, 0x42,
  0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x2e, 0xfe, 0xff, 0xff,
  0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x02, 0x3c, 0x00, 0x00, 0x00, 0x9c, 0xfd, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xce, 0x25, 0x0b, 0x39, 0x0c, 0x00, 0x00, 0x00,
  0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x32, 0x2f, 0x62, 0x69, 0x61, 0x73,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x86, 0xfe, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x48, 0x00, 0x00, 0x00,
  0xf4, 0xfd, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xf9, 0xd8, 0x2e, 0x3c,
  0x19, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69,
  0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x32, 0x2f, 0x4d,
  0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xee, 0xfe, 0xff, 0xff,
  0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x02, 0x3c, 0x00, 0x00, 0x00, 0x5c, 0xfe, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xe5, 0xca, 0x12, 0x39, 0x0c, 0x00, 0x00, 0x00,
  0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31, 0x2f, 0x62, 0x69, 0x61, 0x73,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x46, 0xff, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x48, 0x00, 0x00, 0x00,
  0xb4, 0xfe, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x6c, 0xdf, 0x14, 0x3c,
  0x19, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69,
  0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31, 0x2f, 0x4d,
  0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xae, 0xff, 0xff, 0xff,
  0x14, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x02, 0x34, 0x00, 0x00, 0x00, 0x1c, 0xff, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x32, 0x24, 0xec, 0x38, 0x0a, 0x00, 0x00, 0x00, 0x64, 0x65, 0x6e, 0x73,
  0x65, 0x2f, 0x62, 0x69, 0x61, 0x73, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x14, 0x00,
  0x13, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x40, 0x00, 0x00, 0x00, 0x7c, 0xff, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0xd3, 0xd1, 0x95, 0x3b, 0x17, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75,
  0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65,
  0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x1c, 0x00,
  0x18, 0x00, 0x17, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x4c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x64, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x0c, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xfb, 0xbf, 0xc9, 0x3c, 0x1d, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x72, 0x76, 0x69, 0x6e, 0x67, 0x5f, 0x64, 0x65, 0x66, 0x61,
  0x75, 0x6c, 0x74, 0x5f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x69, 0x6e,
  0x70, 0x75, 0x74, 0x3a, 0x30, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09
};
unsigned int custom_model_tflite_len = 2384;
