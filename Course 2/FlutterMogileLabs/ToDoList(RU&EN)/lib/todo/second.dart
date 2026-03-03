import 'package:flutter/material.dart';
import 'package:mobile_apps/generated/l10n.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:mobile_apps/todo/pages/home.dart';
import 'package:mobile_apps/todo/pages/main_screen.dart';


void main() {
  runApp(const App());
}

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  // Переменная локали
  var _locale = S.en;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        colorSchemeSeed: Colors.deepOrange,
      ),
      supportedLocales: S.supportedLocales,
      locale: _locale,
      localizationsDelegates: const [
          S.delegate,
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],

      builder: (context, child) => Material(
        child: SafeArea(
          child: Stack(
            children: [
              child ?? const SizedBox.shrink(),
              Align(
                alignment: Alignment.topRight,
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: InkResponse(
                    child: Text(
                      _locale.languageCode.toUpperCase(),
                      style: const TextStyle(fontSize: 24, color: Colors.white),
                    ),
                    onTap: () {
                      // Проверяем текущую локаль и изменяем на противоположную
                      final newLocale = S.isEn(_locale) ? S.ru : S.en;
                      setState(() => _locale = newLocale);
                    },
                  ),
                ),
              )
            ],
          ),
        ),
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => MainScreen(),
        '/to_do_list': (context) => Home(),
      },
    );
  }
}