`timescale 1ns / 1ps

module bgr_to_rgb_stream (
    input  wire [23:0] s_pixel,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire        s_sof,
    input  wire        s_eol,

    output wire [23:0] m_pixel,
    output wire        m_valid,
    input  wire        m_ready,
    output wire        m_sof,
    output wire        m_eol
);

    // 这个版本不缓存像素：下游能接收，输入端才报告 ready
    assign s_ready = m_ready;
    assign m_valid = s_valid;

    // BGR 转 RGB
    assign m_pixel = {
        s_pixel[7:0],       // R
        s_pixel[15:8],      // G
        s_pixel[23:16]      // B
    };

    // 标记随对应像素一起通过
    assign m_sof = s_sof;
    assign m_eol = s_eol;

endmodule