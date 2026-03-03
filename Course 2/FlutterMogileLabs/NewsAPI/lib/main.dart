import 'package:flutter/material.dart';
import 'package:news_api_flutter_package/model/article.dart';
import 'package:news_api_flutter_package/news_api_flutter_package.dart';
import 'package:lab5/news.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'My news lab',
      theme: ThemeData(
      colorSchemeSeed: Colors.deepOrange,
    ),
      darkTheme: ThemeData(
        colorScheme: const ColorScheme.dark(),
        useMaterial3: true,
      ),
      home: const MyHomePage(
        title: Text('My news lab',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 30,
            )),
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final Widget title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late Future<List<Article>> future;

  String _search = "Nothing";

  @override
  void initState() {
    future = getNewsData();

    super.initState();
  }

  Future<List<Article>> getNewsData() async {
    NewsAPI newsAPI = NewsAPI("582cf983472c4252a98a77a862d69f82");
    return await newsAPI.getEverything(
      query: _search,
      pageSize: 10,
    );
  }

  Widget buildNewsItem(Article article) {
    return InkWell(
      onTap: () {
        Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) =>
                  NewsWebView(url: article.url!, title: article.title!),
            ));
      },
      child: Card(
        elevation: 4,
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                height: 80,
                width: 80,
                child: Image.network(
                  article.urlToImage ?? "",
                  fit: BoxFit.fitHeight,
                  errorBuilder: (context, error, stackTrace) {
                    return const Icon(Icons.image_not_supported);
                  },
                ),
              ),
              const SizedBox(width: 20),
              Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        article.title!,
                        maxLines: 2,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                        ),
                      ),
                      Text(
                        article.source.name!,
                        style: const TextStyle(color: Colors.grey),
                      ),
                    ],
                  ))
            ],
          ),
        ),
      ),
    );
  }

  Widget buildNewsListView(List<Article> articleList) {
    return ListView.builder(
      itemBuilder: (context, index) {
        Article article = articleList[index];
        return buildNewsItem(article);
      },
      itemCount: articleList.length,
    );
  }

  update(String input){
    setState(() {
      _search = input;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[900],
      appBar: AppBar(
        backgroundColor: Colors.deepOrange,
        title: Align(
          alignment: Alignment.center,
          child: TextField(
            style: const TextStyle(color: Colors.black, fontSize: 20),
            decoration: const InputDecoration(
              hintText: 'Search...',
              hintStyle: TextStyle(color: Colors.black, fontSize: 20),
              contentPadding: EdgeInsets.symmetric(vertical: 15, horizontal: 10), // Adjust the padding
              isDense: true,
              floatingLabelBehavior: FloatingLabelBehavior.always,
            ),
            onSubmitted: (context) {
              setState(() {
                _search = context;
                future = getNewsData();
              });
            },
          ),
        ),
      ),
      body: SafeArea(
          child: Column(
            children: [
              Expanded(
                child: FutureBuilder(
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Center(
                        child: CircularProgressIndicator(),
                      );
                    } else if (snapshot.hasError) {
                      return const Center(
                        child: Text("Error"),
                      );
                    } else {
                      if (snapshot.hasData && snapshot.data!.isNotEmpty) {
                        return buildNewsListView(snapshot.data as List<Article>);
                      } else {
                        return const Center(
                          child: Text("There are not any news"),
                        );
                      }
                    }
                  },
                  future: future,
                ),
              )
            ],
          )),
    );
  }
}