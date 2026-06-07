// Only use DHAT if the feature is enabled
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use egglog_experimental::app::OracleApp;

fn main() {
    // 1. Initialize DHAT (if feature enabled)
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    // 2. Initialize Tracy (if feature enabled)
    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::Registry;
        let tracy_layer = tracing_tracy::TracyLayer::default();
        tracing::subscriber::set_global_default(
            Registry::default().with(tracy_layer)
        ).expect("Failed to set up Tracy");
    }

    // 3. Run Application
    let app = OracleApp::new();
    app.run();

    #[cfg(feature = "dhat-heap")]
    println!("Generating DHAT report...");

    #[cfg(feature = "tracy")]
    {
        println!("Tracy trace is ready!");
        println!("Please open the Tracy GUI and connect to the running process.");
        println!("Press ENTER here in the terminal to exit and close the trace...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
    }
}