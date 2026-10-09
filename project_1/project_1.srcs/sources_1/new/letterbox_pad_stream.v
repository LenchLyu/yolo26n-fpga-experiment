`timescale 1ns / 1ps

module letterbox_pad_stream #(
    parameter [15:0] TARGET_WIDTH  = 16'd640,
    parameter [15:0] TARGET_HEIGHT = 16'd640
) (
    input  wire        clk,
    input  wire        rst_n,

    // PS 配置：缩放后的内容尺寸及左上补边量
    input  wire        cfg_start,
    output wire        cfg_ready,
    input  wire [15:0] cfg_resized_width,
    input  wire [15:0] cfg_resized_height,
    input  wire [15:0] cfg_pad_left,
    input  wire [15:0] cfg_pad_top,

    // 输入流：已经缩放好的 BGR888 内容
    input  wire [23:0] s_pixel_bgr,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire        s_sof,
    input  wire        s_eol,

    // 输出流：固定尺寸 RGB888 画布
    output wire [23:0] m_pixel_rgb,
    output wire        m_valid,
    input  wire        m_ready,
    output wire        m_sof,
    output wire        m_eol,

    output reg         frame_done,
    output reg         error
);

    reg        active;
    reg [15:0] resized_width_reg;
    reg [15:0] resized_height_reg;
    reg [15:0] pad_left_reg;
    reg [15:0] pad_top_reg;

    reg [15:0] out_x;
    reg [15:0] out_y;

    wire [16:0] cfg_right_edge;
    wire [16:0] cfg_bottom_edge;
    wire [16:0] right_edge;
    wire [16:0] bottom_edge;
    wire [16:0] out_x_ext;
    wire [16:0] out_y_ext;

    wire inside_content;
    wire expected_s_sof;
    wire expected_s_eol;
    wire m_fire;

    assign cfg_ready = ~active;

    assign cfg_right_edge =
        {1'b0, cfg_pad_left} + {1'b0, cfg_resized_width};
    assign cfg_bottom_edge =
        {1'b0, cfg_pad_top} + {1'b0, cfg_resized_height};

    assign right_edge =
        {1'b0, pad_left_reg} + {1'b0, resized_width_reg};
    assign bottom_edge =
        {1'b0, pad_top_reg} + {1'b0, resized_height_reg};

    assign out_x_ext = {1'b0, out_x};
    assign out_y_ext = {1'b0, out_y};

    // 当前输出坐标是否落在有效图像内容区域内
    assign inside_content =
        active &&
        (out_x_ext >= {1'b0, pad_left_reg}) &&
        (out_x_ext < right_edge) &&
        (out_y_ext >= {1'b0, pad_top_reg}) &&
        (out_y_ext < bottom_edge);

    // 补边位置由本模块生成；内容区域才接收输入像素
    assign s_ready = inside_content && m_ready;

    // 补边像素可直接生成；内容像素要等输入有效
    assign m_valid = active && (inside_content ? s_valid : 1'b1);

    // 输入是 BGR888，内容像素在此交换 R、B 通道。
    // 114 灰色三个通道相同，补边无需交换。
    assign m_pixel_rgb = inside_content
        ? {s_pixel_bgr[7:0], s_pixel_bgr[15:8], s_pixel_bgr[23:16]}
        : 24'h727272;

    // 输出帧标记针对完整的目标画布生成
    assign m_sof =
        m_valid && (out_x == 16'd0) && (out_y == 16'd0);

    assign m_eol =
        m_valid && (out_x == TARGET_WIDTH - 16'd1);

    assign m_fire = m_valid && m_ready;

    // 检查输入缩放图像应有的 SOF 和每行 EOL 位置
    assign expected_s_sof =
        inside_content &&
        (out_x == pad_left_reg) &&
        (out_y == pad_top_reg);

    assign expected_s_eol =
        inside_content &&
        (out_x_ext == right_edge - 17'd1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active             <= 1'b0;
            resized_width_reg  <= 16'd0;
            resized_height_reg <= 16'd0;
            pad_left_reg       <= 16'd0;
            pad_top_reg        <= 16'd0;
            out_x              <= 16'd0;
            out_y              <= 16'd0;
            frame_done         <= 1'b0;
            error              <= 1'b0;
        end else begin
            frame_done <= 1'b0;

            // 空闲时接收一组新配置
            if (cfg_start && cfg_ready) begin
                if ((TARGET_WIDTH == 16'd0) ||
                    (TARGET_HEIGHT == 16'd0) ||
                    (cfg_resized_width == 16'd0) ||
                    (cfg_resized_height == 16'd0) ||
                    (cfg_right_edge > {1'b0, TARGET_WIDTH}) ||
                    (cfg_bottom_edge > {1'b0, TARGET_HEIGHT})) begin
                    error <= 1'b1;
                end else begin
                    resized_width_reg  <= cfg_resized_width;
                    resized_height_reg <= cfg_resized_height;
                    pad_left_reg       <= cfg_pad_left;
                    pad_top_reg        <= cfg_pad_top;
                    out_x              <= 16'd0;
                    out_y              <= 16'd0;
                    active             <= 1'b1;
                    error              <= 1'b0;
                end
            end

            // 仅在输出像素完成握手时推进目标画布坐标
            if (m_fire) begin
                if (inside_content &&
                    ((s_sof != expected_s_sof) ||
                     (s_eol != expected_s_eol))) begin
                    error <= 1'b1;
                end

                if (out_x == TARGET_WIDTH - 16'd1) begin
                    out_x <= 16'd0;

                    if (out_y == TARGET_HEIGHT - 16'd1) begin
                        out_y      <= 16'd0;
                        active     <= 1'b0;
                        frame_done <= 1'b1;
                    end else begin
                        out_y <= out_y + 16'd1;
                    end
                end else begin
                    out_x <= out_x + 16'd1;
                end
            end
        end
    end

endmodule
