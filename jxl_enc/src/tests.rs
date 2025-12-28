// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Test module for the JPEG XL encoder.

mod bit_writer_tests {
    use crate::bit_writer::BitWriter;

    #[test]
    fn test_roundtrip_simple() {
        let mut writer = BitWriter::new();
        writer.write(8, 0xAB).unwrap();
        writer.write(8, 0xCD).unwrap();
        let bytes = writer.finish();
        assert_eq!(bytes, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_roundtrip_unaligned() {
        let mut writer = BitWriter::new();
        writer.write(3, 0b101).unwrap();
        writer.write(5, 0b10101).unwrap();
        writer.write(4, 0b1111).unwrap();
        writer.zero_pad_to_byte();

        let bytes = writer.finish();
        // 101 + 10101 = 10101_101 = 0xAD
        // 1111 + 0000 = 0000_1111 = 0x0F
        assert_eq!(bytes, vec![0xAD, 0x0F]);
    }
}

mod header_tests {
    use crate::bit_writer::BitWriter;
    use crate::headers::FileHeader;

    #[test]
    fn test_minimal_header() {
        let header = FileHeader::new_rgb(64, 64);
        let mut writer = BitWriter::new();
        header.write(&mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        // Should start with JXL signature
        assert_eq!(&bytes[0..2], &[0xFF, 0x0A]);
        // Should be reasonably small for a default header
        assert!(bytes.len() < 20);
    }
}
