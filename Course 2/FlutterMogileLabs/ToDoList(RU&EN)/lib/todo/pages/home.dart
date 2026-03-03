import 'dart:async';
import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:mobile_apps/generated/l10n.dart';


class DBHelper {
  DBHelper._privateConstructor();
  static final DBHelper instance = DBHelper._privateConstructor();
  static Database? _database;

  Future<Database> getDatabase() async {
    if (_database != null) {
      return _database!;
    } else {
      _database = await _initDB();
      return _database!;
    }
  }

  Future<Database> _initDB() async {
    var documentsDirectory = await getApplicationDocumentsDirectory();
    var path = join(documentsDirectory.path, "ToDoDB.db");
    return await openDatabase(path, version: 1, onCreate: _onCreate);
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute("CREATE TABLE IF NOT EXISTS todo("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "desc TEXT, "
        "done INTEGER)");
  }

  Future<int> insertTodo(String desc) async {
    Database db = await getDatabase();
    return await db.insert('todo', {'desc': desc, 'done': 0});
  }

  Future<List<Map<String, dynamic>>> getTodos() async {
    Database db = await getDatabase();
    return await db.query('todo');
  }

  Future<int> updateTodo(int newID, String new_desc, int newDone) async {
    Database db = await getDatabase();
    return await db.update(
      'todo',
      {'desc': new_desc, 'done': newDone},
      where: 'id = ?',
      whereArgs: [newID],
    );
  }

  Future<int> deleteTodo(int id) async {
    Database db = await getDatabase();
    return await db.delete('todo', where: 'id = ?', whereArgs: [id]);
  }
}


class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}


class _HomeState extends State<Home> {
  String _user_entry = '';
  List<Map<String, dynamic>> todoList = [];

  @override
  void initState() {
    super.initState();
    DBHelper.instance.getTodos().then((todos) {
      setState(() {
        todoList.addAll(todos);
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[900],
      appBar: AppBar(
        title: Text(S.of(context).HomeAppBar),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(8),
        child: ListView.builder(
          itemCount: todoList.length,
          itemBuilder: (BuildContext context, int index) {
            return Container(
              margin: const EdgeInsets.only(bottom: 8),
              child: CheckboxListTile(
                tileColor: Colors.white,
                activeColor: Colors.black54,
                title: Text(
                  todoList[index]['desc'],
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w500,
                    decoration: todoList[index]['done'] != 0
                        ? TextDecoration.lineThrough
                        : TextDecoration.none,
                  ),
                ),
                value: todoList[index]['done'] == 1,
                onChanged: (bool? checked) async {
                  int doneValue = checked! ? 1 : 0;
                  await DBHelper.instance.updateTodo(todoList[index]['id'], todoList[index]['desc'], doneValue);

                  List<Map<String, dynamic>> refreshedTodos = await DBHelper.instance.getTodos();
                  setState(() {
                    todoList = refreshedTodos;
                  });
                },
                secondary: IconButton(
                  icon: const Icon(Icons.delete_forever),
                  color: Colors.black,
                  onPressed: () async {
                    await DBHelper.instance.deleteTodo(todoList[index]['id']);
                    List<Map<String, dynamic>> refreshedTodos = await DBHelper.instance.getTodos();
                    setState(() {
                      todoList = refreshedTodos;
                    });
                  },
                ),
              ),
            );
          },
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          showDialog(context: context, builder: (BuildContext context) {
            return AlertDialog(
              title: Text(S.of(context).SubText),
              content: TextField(
                onChanged: (String value) {
                  _user_entry = value;
                },
              ),
              actions: [
                ElevatedButton(
                  onPressed: () async {
                    if ((_user_entry != null) && (_user_entry != '')) {
                      await DBHelper.instance.insertTodo(_user_entry);
                      List<Map<String, dynamic>> refreshedTodos = await DBHelper.instance.getTodos();
                      setState(() {
                        todoList = refreshedTodos;
                      });
                      _user_entry = '';
                      Navigator.of(context).pop();
                    }
                  },
                  child: Text(S.of(context).Add),
                ),
              ],
            );
          });
        },
        backgroundColor: Colors.white,
        child: const Icon(Icons.add, size: 46, color: Colors.deepOrange),
      ),
    );
  }
}