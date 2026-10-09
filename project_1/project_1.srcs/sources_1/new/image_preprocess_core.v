`timescale 1ns / 1ps
module image_preprocess_core(
    //我们假设输入像素: BGR888，按照[23:16]=B,[15:8]=G,[7:0]=R 来排列
    input  wire [23:0] s_pixel_bgr,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire        s_sof,
    input  wire        s_eof,
    // 输出像素：RGB888，按 [23:16]=R、[15:8]=G、[7:0]=B 排列
    output wire [23:0] m_pixel_rgb,
    output wire        m_valid,
    input  wire        m_ready,
    output wire        m_sof,
    output wire        m_eof
    //s_ 表示输入端，m_ 表示输出端。
    //s_valid：输入端当前有有效像素。
    //s_ready：你的模块准备好接收像素。
    //m_valid：你的模块当前有有效输出像素。
    //m_ready：下游模块准备好接收像素。
    );
    //原样传递握手信号
    assign m_valid     = s_valid;
    assign m_ready     = s_ready;
    //像素顺序重新排列
    assign m_pixel_rgb = {
        s_pixel_bgr[ 7: 0],
        s_pixel_bgr[15: 8],
        s_pixel_bgr[31:16]
    };
    assign m_sof = s_sof;
    assign m_eof = s_eof;
endmodule
