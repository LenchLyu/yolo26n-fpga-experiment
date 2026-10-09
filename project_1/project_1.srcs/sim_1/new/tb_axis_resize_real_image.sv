`timescale 1ns / 1ps

module tb_axis_resize_real_image;

    // 输入图片：1280x720 RGB888
    localparam integer SRC_WIDTH       = 1280;
    localparam integer SRC_HEIGHT      = 720;
    localparam integer SRC_PIXELS      = SRC_WIDTH * SRC_HEIGHT;

    // 输出图片：640x640
    localparam integer OUTPUT_WIDTH    = 640;
    localparam integer OUTPUT_HEIGHT   = 640;
    localparam integer OUTPUT_PIXELS   = OUTPUT_WIDTH * OUTPUT_HEIGHT;

    // 当前 RTL 的图像区为 640x360，上下各补 140 行黑边
    localparam integer PAD_TOP          = 140;
    localparam integer RESIZED_HEIGHT   = 360;
    localparam integer PAD_BOTTOM_START =
        (PAD_TOP + RESIZED_HEIGHT) * OUTPUT_WIDTH;

    // 100 MHz 时钟，周期 10 ns
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst_n;

    // 输入 AXI-Stream
    reg  [23:0] s_axis_tdata;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    reg         s_axis_tuser;
    reg         s_axis_tlast;

    // 输出 AXI-Stream
    wire [23:0] m_axis_tdata;
    wire        m_axis_tvalid;
    reg         m_axis_tready;
    wire        m_axis_tuser;
    wire        m_axis_tlast;

    // 一帧输入像素。每个元素是一个 24 位 RRGGBB 像素。
    reg [23:0] pixels [0:SRC_PIXELS-1];

    string input_hex_path;
    string output_hex_path;

    integer input_file_check;
    integer output_file;

    integer out_count;
    integer sof_count;
    integer eol_count;
    integer error_count;

    integer x;
    integer y;
    integer pixel_index;
    integer timeout_count;

    // 被测 RTL 模块
    axis_resize_letterbox_1280x720_to_640x640 dut (
        .clk           (clk),
        .rst_n          (rst_n),

        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .s_axis_tuser   (s_axis_tuser),
        .s_axis_tlast   (s_axis_tlast),

        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tready  (m_axis_tready),
        .m_axis_tuser   (m_axis_tuser),
        .m_axis_tlast   (m_axis_tlast)
    );

    // 监视输出握手。只有 valid 和 ready 同时为 1 时，像素才算输出。
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (out_count >= OUTPUT_PIXELS) begin
                if (error_count < 10) begin
                    $display(
                        "ERROR: Extra output pixel detected at count=%0d",
                        out_count
                    );
                end
                error_count = error_count + 1;
            end else begin
                // SOF 只能出现在整帧的第一个输出像素。
                if (out_count == 0) begin
                    if (m_axis_tuser !== 1'b1) begin
                        if (error_count < 10)
                            $display("ERROR: m_axis_tuser was not asserted for the first output pixel.");
                        error_count = error_count + 1;
                    end
                end else begin
                    if (m_axis_tuser !== 1'b0) begin
                        if (error_count < 10)
                            $display("ERROR: m_axis_tuser was asserted after the first output pixel.");
                        error_count = error_count + 1;
                    end
                end

                if (m_axis_tuser === 1'b1)
                    sof_count = sof_count + 1;

                // 每 640 个输出像素应有一个 TLAST。
                if ((out_count % OUTPUT_WIDTH) == (OUTPUT_WIDTH - 1)) begin
                    if (m_axis_tlast !== 1'b1) begin
                        if (error_count < 10)
                            $display(
                                "ERROR: m_axis_tlast was not asserted at output pixel %0d",
                                out_count
                            );
                        error_count = error_count + 1;
                    end
                end else begin
                    if (m_axis_tlast !== 1'b0) begin
                        if (error_count < 10)
                            $display(
                                "ERROR: Unexpected m_axis_tlast at output pixel %0d",
                                out_count
                            );
                        error_count = error_count + 1;
                    end
                end

                if (m_axis_tlast === 1'b1)
                    eol_count = eol_count + 1;

                // 顶部和底部补边区应该全是黑色。
                if ((out_count < PAD_TOP * OUTPUT_WIDTH) ||
                    (out_count >= PAD_BOTTOM_START)) begin
                    if (m_axis_tdata !== 24'h727272) begin
                        if (error_count < 10)
                            $display(
                                "ERROR: Padding pixel is not gray: index=%0d data=%06h",
                                out_count,
                                m_axis_tdata
                            );
                        error_count = error_count + 1;
                    end
                end
            end

            // 每个输出握手写一行 RGB888：RRGGBB。
            $fwrite(output_file, "%06h\n", m_axis_tdata);

            out_count = out_count + 1;
        end
    end

    // 仿真主流程
    initial begin
        rst_n         = 1'b0;
        s_axis_tdata  = 24'h000000;
        s_axis_tvalid = 1'b0;
        s_axis_tuser  = 1'b0;
        s_axis_tlast  = 1'b0;
        m_axis_tready = 1'b1;

        out_count   = 0;
        sof_count   = 0;
        eol_count   = 0;
        error_count = 0;

        /*
         * XSim 默认工作目录通常是：
         * <工程目录>/<工程名>.sim/sim_1/behav/xsim
         *
         * 从该目录向上四级可回到 Vivado 工程目录。
         * 克隆 GitHub 仓库后，只要保留 Vivado 的常见目录结构，
         * 这两个相对路径就不需要随电脑路径改变。
         *
         * 也可以通过 INPUT_HEX 和 OUTPUT_HEX 仿真参数覆盖它们。
         */
        input_hex_path  = "../../../../testdata/input_rgb.hex";
        output_hex_path = "../../../../testdata/sim_output_rgb.hex";

        if ($value$plusargs("INPUT_HEX=%s", input_hex_path))
            $display("Using input file argument: %s", input_hex_path);
        else
            $display("Using default input file: %s", input_hex_path);

        if ($value$plusargs("OUTPUT_HEX=%s", output_hex_path))
            $display("Using output file argument: %s", output_hex_path);
        else
            $display("Using default output file: %s", output_hex_path);

        // 先检查输入文件是否能打开。
        input_file_check = $fopen(input_hex_path, "r");
        if (input_file_check == 0) begin
            $fatal(
                1,
                "Cannot find input file: %s. Run image_to_hex.py or check the XSim relative path.",
                input_hex_path
            );
        end
        $fclose(input_file_check);

        // 读取 Python 生成的 921600 个 RGB 像素。
        $readmemh(input_hex_path, pixels);

        // 检查输入数据是否成功读入。
        if ((^pixels[0]) === 1'bx) begin
            $fatal(1, "Input hex file was not loaded correctly; pixels[0] contains unknown bits.");
        end

        // 创建仿真输出文件。
        output_file = $fopen(output_hex_path, "w");
        if (output_file == 0) begin
            $fatal(
                1,
                "Cannot create simulation output file: %s. Check that the directory exists and is writable.",
                output_hex_path
            );
        end

        // 保持复位 5 个时钟周期。
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        $display("Sending %0d input pixels.", SRC_PIXELS);

        pixel_index = 0;

        // 按行发送图像。valid 在 ready 到来之前保持，符合 AXI-Stream 握手。
        for (y = 0; y < SRC_HEIGHT; y = y + 1) begin
            for (x = 0; x < SRC_WIDTH; x = x + 1) begin
                @(negedge clk);

                s_axis_tdata  = pixels[pixel_index];
                s_axis_tvalid = 1'b1;
                s_axis_tuser  = (pixel_index == 0);
                s_axis_tlast  = (x == SRC_WIDTH - 1);

                // 等待一次输入握手：valid && ready。
                @(posedge clk);
                while (s_axis_tready !== 1'b1)
                    @(posedge clk);

                pixel_index = pixel_index + 1;
            end
        end

        // 最后一个像素握手完成后，撤销输入控制信号。
        @(negedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tuser  = 1'b0;
        s_axis_tlast  = 1'b0;
        s_axis_tdata  = 24'h000000;

        $display("Input frame sent; waiting for the output frame to complete.");

        // 等待完整的 640x640 输出帧。
        wait (out_count >= OUTPUT_PIXELS);
        @(negedge clk);

        $fclose(output_file);

        $display("Output pixel count: %0d; expected: %0d", out_count, OUTPUT_PIXELS);
        $display("SOF count: %0d; expected: 1", sof_count);
        $display("EOL count: %0d; expected: %0d", eol_count, OUTPUT_HEIGHT);
        $display("Detected error count: %0d", error_count);

        if (out_count != OUTPUT_PIXELS)
            $fatal(1, "FAIL: Incorrect output pixel count.");

        if (sof_count != 1)
            $fatal(1, "FAIL: Incorrect SOF count.");

        if (eol_count != OUTPUT_HEIGHT)
            $fatal(1, "FAIL: Incorrect end-of-line marker count.");

        if (error_count != 0)
            $fatal(1, "FAIL: The testbench detected errors.");

        $display("PASS: Pixel count, SOF, TLAST, and gray padding checks passed.");
        $display("Simulation output file: %s", output_hex_path);

        $finish;
    end

    // 防止 ready 一直不来或输出帧一直不结束。
    initial begin
        #30000000;
        $fatal(1, "TIMEOUT: Simulation did not complete within 30 ms.");
    end

endmodule
