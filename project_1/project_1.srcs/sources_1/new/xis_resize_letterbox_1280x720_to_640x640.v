`timescale 1ns / 1ps

module axis_resize_letterbox_1280x720_to_640x640 #(
    parameter integer SRC_WIDTH     = 1280,
    parameter integer SRC_HEIGHT    = 720,
    parameter integer TARGET_HEIGHT = 640
) (
    input  wire        clk,
    input  wire        rst_n,

    // 输入：VDMA MM2S 输出的 RGB888 AXI4-Stream
    input  wire [23:0] s_axis_tdata,  // 从ddr中取走的像素
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tuser,  // SOF：帧开始
    input  wire        s_axis_tlast,  // EOL：行结束

    // 输出：640×640 RGB888 AXI4-Stream
    output wire [23:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tuser,  // SOF
    output wire        m_axis_tlast   // EOL
);

    // 第一版只做固定 2 倍缩小：
    // 1280×720 -> 640×360，再上下补边到 640×640。
    localparam integer RESIZED_WIDTH  = SRC_WIDTH / 2;
    localparam integer RESIZED_HEIGHT = SRC_HEIGHT / 2;
    localparam integer PAD_TOP =
        (TARGET_HEIGHT - RESIZED_HEIGHT) / 2;

    localparam [1:0] ST_WAIT       = 2'd0;
    localparam [1:0] ST_PAD_TOP    = 2'd1;
    localparam [1:0] ST_IMAGE      = 2'd2;
    localparam [1:0] ST_PAD_BOTTOM = 2'd3;

    reg [1:0] state;

    // 输入图像当前像素坐标
    reg [15:0] src_x;
    reg [15:0] src_y;

    // 输出画布当前像素坐标
    reg [15:0] dst_x;
    reg [15:0] dst_y;

    wire selected_pixel;
    wire input_fire;
    wire output_fire;

    // 最近邻 2 倍缩小：只保留偶数行、偶数列的像素。
    // 例如保留输入坐标 (0,0)、(2,0)、(4,0)……
    assign selected_pixel =
        (src_x[0] == 1'b0) &&
        (src_y[0] == 1'b0);

    // 只有在处理图像时才接收输入。
    // 被丢弃的像素不需要等待下游；被保留的像素要等下游 ready。
    assign s_axis_tready =
        (state == ST_IMAGE) ?
        ((!selected_pixel) || m_axis_tready) :
        1'b0;

    assign input_fire = s_axis_tvalid && s_axis_tready;
    assign output_fire = m_axis_tvalid && m_axis_tready;

    // 上下补边时生成黑色像素；图像区域转发选中的输入像素。
    assign m_axis_tvalid =
        ((state == ST_PAD_TOP) || (state == ST_PAD_BOTTOM)) ?
            1'b1 :
        ((state == ST_IMAGE) &&
         selected_pixel &&
         s_axis_tvalid);

    assign m_axis_tdata =
        (state == ST_IMAGE) ? s_axis_tdata : 24'h727272;

    // 输出画布第一个像素标记 SOF，每行最后一个像素标记 EOL。
    assign m_axis_tuser =
        m_axis_tvalid && (dst_x == 0) && (dst_y == 0);

    assign m_axis_tlast =
        m_axis_tvalid && (dst_x == RESIZED_WIDTH - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_WAIT;

            src_x <= 0;
            src_y <= 0;
            dst_x <= 0;
            dst_y <= 0;
        end else begin
            case (state)

                // 等待 VDMA 给出一帧的开始标记。
                // 此时不接收输入，VDMA 会保持 SOF 像素不变。
                ST_WAIT: begin
                    if (s_axis_tvalid && s_axis_tuser) begin
                        src_x <= 0;
                        src_y <= 0;
                        dst_x <= 0;
                        dst_y <= 0;
                        state <= ST_PAD_TOP;
                    end
                end

                // 输出顶部黑边。
                ST_PAD_TOP: begin
                    if (output_fire) begin
                        if (dst_x == RESIZED_WIDTH - 1) begin
                            dst_x <= 0;

                            if (dst_y == PAD_TOP - 1) begin
                                dst_y <= PAD_TOP;
                                src_x <= 0;
                                src_y <= 0;
                                state <= ST_IMAGE;
                            end else begin
                                dst_y <= dst_y + 1'b1;
                            end
                        end else begin
                            dst_x <= dst_x + 1'b1;
                        end
                    end
                end

                // 接收源图像。只有偶数行、偶数列输出。
                ST_IMAGE: begin
                    if (input_fire) begin
                        if (selected_pixel) begin
                            if (dst_x == RESIZED_WIDTH - 1) begin
                                dst_x <= 0;
                                dst_y <= dst_y + 1'b1;
                            end else begin
                                dst_x <= dst_x + 1'b1;
                            end
                        end

                        // 输入 TLAST 表示一行结束。
                        if (s_axis_tlast) begin
                            src_x <= 0;

                            if (src_y == SRC_HEIGHT - 1) begin
                                src_y <= 0;
                                state <= ST_PAD_BOTTOM;
                            end else begin
                                src_y <= src_y + 1'b1;
                            end
                        end else begin
                            src_x <= src_x + 1'b1;
                        end
                    end
                end

                // 输入图像处理完毕，输出底部黑边。
                ST_PAD_BOTTOM: begin
                    if (output_fire) begin
                        if (dst_x == RESIZED_WIDTH - 1) begin
                            dst_x <= 0;

                            if (dst_y == TARGET_HEIGHT - 1) begin
                                dst_x <= 0;
                                dst_y <= 0;
                                state <= ST_WAIT;
                            end else begin
                                dst_y <= dst_y + 1'b1;
                            end
                        end else begin
                            dst_x <= dst_x + 1'b1;
                        end
                    end
                end

                default: begin
                    state <= ST_WAIT;
                end
            endcase
        end
    end

endmodule