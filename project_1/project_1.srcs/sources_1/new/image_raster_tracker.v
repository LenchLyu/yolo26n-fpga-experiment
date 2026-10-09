`timescale 1ns / 1ps
//本模块是一个行列坐标计数器：观察像素流，根据 SOF、EOL 和握手状态，输出当前像素的 (x, y) 坐标

module image_raster_tracker (
    input  wire        clk,
    input  wire        rst_n,
    // 本帧尺寸，整帧传输期间保持不变
    // 一般来说输入的图像width和height不会超过1980，这里采用16位是简化代码以及避免图像像素爆炸
    input  wire [15:0] cfg_width,
    input  wire [15:0] cfg_height,
    // 输入像素流控制信号
    input  wire        s_valid,
    input  wire        s_ready,
    input  wire        s_sof,
    input  wire        s_eol,
    // 当前输入像素的坐标
    output wire [15:0] pixel_x,
    output wire [15:0] pixel_y,
    output wire        coord_valid,
    //coord_valid：坐标是否对应一颗有效输入像素；这里直接等于 s_valid。
    // 最后一行的最后一个像素完成握手时拉高
    output wire        frame_done,
    // 检测到尺寸或 EOL 位置错误后保持为 1，复位清零
    output reg         format_error
    //format_error：发现尺寸为零、坐标越界或 EOL 位置不符后置 1；只能通过复位清零。
);

    reg [15:0] x_count;
    reg [15:0] y_count;
    wire        pixel_fire;
    wire [15:0] current_x;
    wire [15:0] current_y;

    // 只有 s_valid 和 s_ready 同时为 1，计数器才更新。若下游暂停，s_ready=0，即使 s_valid=1，本拍也没有接收像素，计数不前进。
    assign pixel_fire = s_valid && s_ready;

    // SOF 所在的像素就是新帧的 (0, 0)
    //通常 x_count、y_count 已经指向当前像素。若当前有效像素带 SOF，则无论计数器之前是什么值，都把这颗像素视作新帧的 (0,0)。
    assign current_x = (s_valid && s_sof) ? 16'd0 : x_count;
    assign current_y = (s_valid && s_sof) ? 16'd0 : y_count;

    assign pixel_x = current_x;
    assign pixel_y = current_y;
    assign coord_valid = s_valid;

    // 只在整帧最后一颗像素握手成功时拉高
    //条件包括计数器能更新、当前帧为行的末位、当前图片有效、像素计数器==行列的末位
    assign frame_done =
        pixel_fire &&
        s_eol &&
        (cfg_width  != 16'd0) &&
        (cfg_height != 16'd0) &&
        (current_x == cfg_width  - 16'd1) &&
        (current_y == cfg_height - 16'd1);

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin
            x_count     <= 16'd0;
            y_count     <= 16'd0;
            format_error <= 1'b0;
        end else if (pixel_fire) begin

            // 检查尺寸有效，且 EOL 是否出现在本行最后一颗像素上
            if ((cfg_width == 16'd0) ||
                (cfg_height == 16'd0) ||
                (current_x >= cfg_width) ||
                (current_y >= cfg_height) ||
                (s_eol != (current_x == cfg_width - 16'd1))) begin
                format_error <= 1'b1;
            end

            // 接收完一颗像素后，更新下一颗像素的坐标
            //当前行接收完毕，进入下一行的数据接收
            if (s_eol) begin
                x_count <= 16'd0;

                // 最后一行结束后回到 y=0，等待下一帧
                if ((cfg_height != 16'd0) &&
                    (current_y == cfg_height - 16'd1)) begin
                    y_count <= 16'd0;
                end else begin
                    y_count <= current_y + 16'd1;
                end
            end else begin
                x_count <= current_x + 16'd1;
                y_count <= current_y;
            end
        end
    end

endmodule